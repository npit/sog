"""Module for policy functionality
"""
# native
import copy
import logging
import time
import os
import random

from os.path import join

# third-party
import numpy as np
from stable_baselines3.ppo import MlpPolicy, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


import torch

# local
from sog.utils import timestamp

class Policy:
    """Abstract policy
    """
    name = None
    tracker = None

    def train(self, **kwargs):
        pass
    def initialize(self, env, **kwargs):
        pass
    def get_weights(self):
        return 0.0
    def load(self, path):
        pass
    def update_model(self, model):
        raise ValueError(f"Attempted to invoke abstract model updating function!")
        del self.model
        self.model = model



class Random(Policy):
    """Random policy
    """
    name = "random"
    def apply(self, agent, env):
        """Apply the policy to generate an output action

        Args:
            agent (Agent): SOG agent object
            env (SOG): SOG environment object

        Returns:
            (gym.spaces.Dict): A randomly sampled action from the action space
        """
        return env.action_space.sample()

class Honest(Policy):
    """Honest policy that truthfully produces messages and actions
    """

    name = "honest"
    def apply(self, agent, env):
        """Apply the policy to generate an output action

        Args:
            agent (Agent): SOG agent object
            env (SOG): SOG environment object

        Returns:
            (gym.spaces.Dict): A randomly sampled action from the action space
        """
        action = {}
        # messages are true food locations
        other_worlds = env.get_observed_worlds(agent.name)
        true_food_locations = [ow.get_one_hot_food_location() if ow.foods else np.zeros(ow.size_1d) for ow in other_worlds]
        agent.observation_history.append(copy.deepcopy(true_food_locations))
        messages = np.concatenate(true_food_locations)

        # movement
        own_world = [w for w in env.worlds if agent == w.agent][0]
        # consult messages
        received_messages = agent.parse_observation(env.get_incoming_messages(agent.name, concatenate=True))

        # reshape to num_senders x world_dim
        received_messages = received_messages.reshape(-1, own_world.size_1d)

        # head to the argmax of the mean
        movement = received_messages.mean(axis=0).argmax()
        movement_oh = np.eye(agent.world_size)[movement]

        action = np.concatenate((movement_oh, messages))
        return action


class SB_PPO(Policy):

    name = "ppo"
    def __init__(self, **kwargs):
        self.model = None

    def update_model(self, weights_dict):
        self.model.set_parameters(weights_dict)

    def _instantiate_model(self, constructor_lambda, **kwargs):
        self.run_path = kwargs.pop("run_path", timestamp())
        kwargs['batch_size'] = kwargs.get('batch_size', 512)
        kwargs['device'] = kwargs.get("device", "cuda")
        kwargs['verbose'] = kwargs.get('verbose', 2)
        kwargs['seed'] = kwargs.get('seed', 42)

        if callable(constructor_lambda):
            env = constructor_lambda()
        else:
            env = constructor_lambda

        self.model = PPO("MlpPolicy", env, **kwargs)
        self.model.set_logger(configure(join(self.run_path, self.prefix), ['stdout', 'log', 'csv', 'tensorboard']))
        return env

    @staticmethod
    def make_env(constructor_fn, seed, idx, total, path):
        def _init():
            logging.basicConfig(level=logging.INFO, filename=join(path, f'env_{idx}_logfile.log'))
            logging.info(f"Creating environment {idx+1} / {total} with seed: {seed}")
            env = constructor_fn(instance_id=idx)
            env.seed(seed)
            return env
        set_random_seed(seed)
        return _init

    def train(self, constructor_lambda, **kwargs):
        """PPO training function

        Args:
            env (sog.SOG): THe environment object
        """
        num_parallel = kwargs.pop("num_parallel", 2)
        try:
            run_path = self.run_path
        except AttributeError:
            run_path = kwargs.pop("run_path", "./")
        tb_log = os.path.join(run_path, "tb")
        kwargs['tensorboard_log'] = tb_log
        kwargs['seed'] = kwargs.get('seed', 42)
        steps = kwargs.pop("train_timesteps", 2048)
        callbacks = []

        if not torch.cuda.is_available():
            logging.warning("Cuda is not available.")

        if num_parallel > 1:
            # for training, we need the environment lambda for vectorization
            env = SubprocVecEnv([SB_PPO.make_env(constructor_lambda, kwargs['seed'] + i, i, num_parallel, run_path) for i in range(num_parallel)], start_method='forkserver')
            env._instance_id = f"env_multiproc_{num_parallel}"
            # self._instantiate_model(env, run_path=run_path)
            if kwargs.get("sync_parallel_models", True):
                callbacks += [ModelSyncCallback(self.model)] 
            env_config = env.env_method('get_configuration')

        else:
            # else, we need the environment object
            if callable(constructor_lambda):
                env = constructor_lambda()
            else:
                env = constructor_lambda
            env_config = env.get_configuration()

        callbacks += [EvalCallback(env)]
        learner_init_func = kwargs.pop("learner_init_func", lambda: None)
        learner_init_func(env)

        self.model.learn(total_timesteps=steps, callback=callbacks, eval_log_path=self.run_path + f"_eval_{self.prefix}")
        self.model_path = os.path.join(self.run_path, self.name)
        self.model.save(self.model_path)

        if isinstance(env_config, list):
            assert all(x == env_config[0] for x in env_config), "Got different environment configurations after training!"
            env_config = env_config[0]

        return env_config

    def load(self, path):
        self.model.load(path, print_system_info=True)

    def apply(self, agent, env):
        obs = env.get_state(agent.name)
        action, new_env = self.model.predict(obs, env)
        return action

    def initialize(self, env, **kwargs):
        self.prefix = kwargs.pop("prefix", "")
        logging.info(f"Initializing policy of [{self.prefix}] from env: {env._instance_id}")
        path = kwargs.pop("policy_path", None)
        self._instantiate_model(env, **kwargs)
        if path is not None:
            self.load(path)

    def get_weights(self):
        keys_weights = self.model.policy.state_dict()
        return keys_weights['value_net.weight'].reshape(-1).cpu().numpy()
    

class ModelSyncCallback(BaseCallback):
    def __init__(self, model, verbose=0, sync_save_path=None):
        super(ModelSyncCallback, self).__init__(verbose)
        # self.sync_save_path = sync_save_path or "sync_model.bin"

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        Update learning model under training at the beginning of each rollout,
        such that the policy used is up-to-date with current weights.
        """
        self.training_env.env_method('update_learner_model', model_weights=self.model.get_parameters())


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def instantiate(which: str):
    """Policy object factory function

    Args:
        which (str): Requested policy name

    Raises:
        NotImplementedError: On undefined requested policy

    Returns:
        _type_: _description_
    """
    available_policies = (Honest, Random, SB_PPO)
    for po in available_policies:
        if po.name == which:
            return po()
    raise NotImplementedError(f"Undefined policy: {which}")

