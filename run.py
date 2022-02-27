import sys
import os
print(sys.executable, os.path.abspath(sys.executable))
# native
import random
import logging
from argparse import ArgumentParser, Namespace
import json
from os.path import join
import os
from functools import partial

# local
from sog.sog import SOG
from sog.reporting import instantiate as instantiate_tracker
from sog.evaluate import Evaluator
from sog.betrayal import BetrayalDatasetBuilder
from sog.penalization.penalizers import instantiate as instantiate_penalizer
from sog.presets import available_presets
from sog.utils import read_json, timestamp, configure_logging, log_to_stdout, profiling_wrapper, nop_wrapper
from sog.policy import instantiate as instantiate_policy

def parse_args():
    parser = ArgumentParser()
    # environment
    parser.add_argument('--preset', help='Configuration preset to apply.', default="random2", type=str)
    # training
    parser.add_argument('--train', help='Whether to train learning agent', action='store_true')
    parser.add_argument('--learning_agent', help='Specify which agent to train, if any.', type=str, default='learner')
    parser.add_argument('--policy_load_config', help='Json with agent: pretrained policy path to load.', type=str, default="{}")
    parser.add_argument('--train_timesteps', help='Training timesteps.', default=3000, type=int)
    parser.add_argument('--num_episodes', help='Number of episodes for evaluation.', default=50, type=int)
    parser.add_argument('--max_iterations', help='Max number of episode iterations.', default=1e3, type=int)
    parser.add_argument('--num_parallel', help='Number of parallel environments to train.', default=2, type=int)

    # penalization
    parser.add_argument('--penalizer_path', help='Path to trained penalizer to utilize towards discouraging betrayal.', default=None, type=str)
    parser.add_argument('--penalized_agents', help='Agent name to penalize.', default='learner', type=str)
    parser.add_argument('--apply_penalization', help='Whether to apply the computed penalization.', action='store_true', default=False)
    parser.add_argument('--penalization_magnitude', help='Maximum magnitude of penalization to apply, scaled by betrayal esimation. Defaults to maximum food nutrition.', type=float, default=None)

    # environment
    parser.add_argument('--world_size', help='Size for each agent world.', default=8, type=int)
    parser.add_argument('--max_food_per_world', help='Max number of food items per world.', default=None, type=int)
    parser.add_argument('--nutrition_config', help='Nutrition configuration.', default="{}")
    parser.add_argument('--no_eat_penalty', help='Amount of reward for not eating in a timestep.', default=0, type=int)
    parser.add_argument('--hunger_delta', help='Hunger step per iteration without nutrition.', default=0.1, type=float)
    parser.add_argument('--observe_hunger', help='Direct hunger observation.', default=False, action="store_true")
    # misc
    parser.add_argument('--run_name', help='Specify run name, allowing run reinitialization (useful for testing with a single run).', default=None)
    parser.add_argument('--log_level', help='Logging level', default='DEBUG')
    parser.add_argument('--tracker', help='Experiment tracker.', default=None, type=str)
    parser.add_argument('--round_begins_with_messaging', help='Whether to begin each round gathering messages from all agents.', default=False, action="store_true")
    parser.add_argument('--collect_betrayal_data', help='Whether to collect betrayal dataset instances.', default=False, action="store_true")
    parser.add_argument('--profiling', help='Whether to profile program execution.', default=False, action="store_true")
    args = parser.parse_args()

    # arg postprocessing
    args.policy_load_config = read_json(args.policy_load_config)

    return parser, args

def generate_run_id_from_args(parser, args):

    # detect provided args
    class _Placeholder:
        pass
    placeholder = _Placeholder()
    ph_ns = Namespace(**{key: placeholder for key in vars(args)})
    parser.parse_args(namespace=ph_ns)
    explicit_args = [k for k, v in vars(ph_ns).items() if v is not placeholder]
    useful_explicit = "train_timesteps world_size hunger_delta no_eat_penalty nutrition_config max_food_per_world".split()
    explicit_args = [x for x in explicit_args if x in useful_explicit]
    if not explicit_args:
        return None
    abbreviate = lambda key: "".join(x[0] for x in key.split("_"))
    args = {abbreviate(k): v for (k, v) in vars(args).items() if k in explicit_args}
    run_id = "_".join(f"{k}{v}" for (k, v) in list(sorted(args.items(), key=lambda x: x[0])))
    return run_id



def run():
    parser, args = parse_args()
    exp_tracker = instantiate_tracker(which=args.tracker, run_name=args.run_name)

    run_name = args.run_name
    if run_name is None:
        run_name = generate_run_id_from_args(parser, args)

    run_id = exp_tracker.get_run_id()

    run_path = run_name = f"{run_name}_{run_id}"

    # save arguments
    os.makedirs(run_path)
    with open(join(run_path, "args.json"), "w") as f:
        json.dump(vars(args), f)

    exp_tracker.set_run_name(run_path)

    configure_logging(run_path, level=args.log_level)

    execution_wrapper = profiling_wrapper if args.profiling else nop_wrapper

    with execution_wrapper(output_folder=run_path):
        logging.info("Configuration:\n" + json.dumps(args.__dict__, indent=2))

        presets = available_presets[args.preset]

        betrayal_analyzer = BetrayalDatasetBuilder() if args.collect_betrayal_data else None


        # policy initialization
        policy_init_funcs = SOG.get_agent_init_func(presets, run_path, args.policy_load_config)

        # evaluation & ground truth
        tracker_lambda = exp_tracker.get_lambda()
        evaluator = Evaluator(max_iterations=args.max_iterations,
                            num_episodes=args.num_episodes, tracker_lambda=tracker_lambda)

        # penalization
        penalizer = None
        if args.penalizer_path:
            penalizer = instantiate_penalizer(args.penalizer_path)
            assert betrayal_analyzer is not None, "Cannot apply penalizer without a betrayal analyzer."

        constructor_lambda = lambda **kwargs: SOG(
            world_size=args.world_size,
            learning_agent_name=args.learning_agent,
            nutrition_config=args.nutrition_config,
            agent_presets=presets,
            agent_initializer_funcs=policy_init_funcs,
            max_food_per_world=args.max_food_per_world,
            no_eat_penalty=args.no_eat_penalty,
            hunger_delta=args.hunger_delta,
            round_begins_with_messaging=args.round_begins_with_messaging,
            observe_hunger=args.observe_hunger,
            penalized_agents=args.penalized_agents,
            apply_penalization=args.apply_penalization,
            penalizer=penalizer,
            betrayal_analyzer=betrayal_analyzer,
            evaluator=evaluator,
            **kwargs
            )

        if args.train:
            # train
            print("Starting training.")
            logging.info("Starting training.")
            learing_agent_preset = [x for x in presets if x['name'] == args.learning_agent]
            assert len(learing_agent_preset) == 1, f'Multiple learning agent configurations encountered: {learning_agent_preset}'
            learning_agent_preset = learing_agent_preset[0]

            policy = instantiate_policy(learning_agent_preset['policy'])
            def learner_init(env, name=args.learning_agent):
                policy.initialize(env, prefix=args.learning_agent, run_path=run_path, policy_path=args.policy_load_config.get(name))
            env_config = policy.train(constructor_lambda, train_timesteps=args.train_timesteps, callback=exp_tracker.get_callback(), num_parallel=args.num_parallel, run_path=run_path, learner_init_func=learner_init)

            exp_tracker.update_config(env_config)
            exp_tracker.conclude()


            logging.info("Training complete -- shutting down.")
            print(f"Training complete -- logs @ {run_path}")

        else:

            env = constructor_lambda()
            env.reset()
            exp_tracker.update_config(env.get_configuration())

            log_to_stdout()

            logging.info("Beginning loop.")
            logging.info("################")
            for episode in range(args.num_episodes):
                env.reset()
                evaluator.set_episode(episode)

                while env.iteration < args.max_iterations:
                    # observation, reward, done, info = env.last()
                    iteration = env.iteration
                    evaluator.set_iteration(iteration)

                    # learning agent
                    action = env.learning_agent.apply_policy(env)
                    obs, reward_delta, done, info = env.step(action)

                    if done:
                        logging.info(f"All food consumed after {env.iteration} iterations.")
                        break
                evaluator.post_process_episode()
            # aggregate performance across all episodes
            evaluator.post_process_evaluation()

            if args.collect_betrayal_data:
                betrayal_analyzer.flush_container_to_disk(run_path)
            exp_tracker.conclude()

if __name__ == "__main__":
    run()