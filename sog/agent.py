"""Agent functionality"""
import random
from collections import Counter
import logging
import copy
import enum

import numpy as np

from sog.utils import log_store
from sog.policy import instantiate
from sog.food import MAX_HUNGER

class Policies(enum.Enum):
    random = "random"
    honest = "honest"
    ppo = "ppo"

class Agent:
    """An observer / gatherer agent"""

    def __repr__(self):
        return f"agent_object:{self.name}"

    def set_world_size(self, sz: int):
        """World size setter"""
        self.world_size = sz
    def set_observed_world_sizes(self, szs: list):
        """Observed world sizes setter"""
        self.observed_world_sizes = szs

    def get_action(self, env, action_key, previous_action=None):
        """Obtain movement action for the agent

        Args:
            env (SOG): _The SOG environment
            previous_action (gym.Spaces.Dict, optional): _Previous action executed by the agent. Defaults to None.

        Returns:
            gym.Spaces.Discrete: Movement index
        """
        if self.policy_type == "single_global":
            return previous_action[action_key]
        elif self.policy_type == "marginals":
            return self.apply_policy(env, action_key)


    def __init__(self, name, index, policy='random', **kwargs):
        """Constructor

        Args:
          id_ (int): Agent ID
          policy: Input policy model. Defaults to None.
          truthfulness (float, optional): Determines a prior on agent truthfullness.
          If None, behaviour is not predetermined and is entirely learned. Defaults to None.

          kwargs:
            "policy_type": Determines the type of policy the agent has. Can be:
                "single_global": Agent has a single function that outputs both movement and messaging actions.
                "marginals": Agent has multiple policy functions for each type of action (movement, messaging)
        """

        self.id_ = self.name = name
        self.index = index
        self.policy = instantiate(policy)
        self.policy_name = policy

        self.hunger = kwargs.get("starting_hunger", 0)
        self.food_log = []
        self.communication_log = []

        self.message_history = []
        self.observation_history = []
        self.hunger_history = []

        self.policy_type = kwargs.get("policy_type", "single_global")
        self.hunger_config = kwargs.get("hunger_config", {"type": "noise_uniform"})
        self.last_movement = None
        self.last_intended_messages = None


    def increase_hunger(self, delta=1):
        """Hunger step function

        Args:
            delta (int, optional): Decay amount. Defaults to 1.
        """
        self.hunger = min(self.hunger + delta, MAX_HUNGER)
        self.hunger_history.append(self.hunger)

    def consume_food(self, food_item):
        """Function to log food consumption

        Args:
            food_item (Food): A consumed food item
        """
        # always update history
        self.food_log.append(food_item)
        if food_item is not None:
            self.hunger = max(0, self.hunger - food_item.get_nutrition_value())

    def __str__(self):
        return f"{self.name} hunger={self.hunger}"

    def get_policy(self):
        return self.policy

    def get_hunger(self):
        return self.hunger

    def get_intended_messages(self):
        return self.last_intended_messages 

    def parse_action(self, action, as_dict=False):
        action = action.squeeze()
        movement = action[:self.world_size]
        messages = action[self.world_size:]
        if as_dict:
            return {"movement": movement, "messages": messages}
        return movement, messages

    def parse_observation(self, obs):
        return obs

    def apply_policy(self, env, action_key=None):
        """Apply agent policy

        Args:
            env (SOG): The SOG environment object

        Returns:
            gym.Space: The action gym Space
        """
        action = self.policy.apply(self, env)
        movement, messages = self.parse_action(action)

        # action["messages_intended"] = copy.deepcopy(messages)

        # hunger effect
        self.last_intended_messages = copy.deepcopy(messages)
        messages  = self.apply_hunger_effects(messages)
        self.message_history.append(action)

        self.last_movement = movement
        action = np.concatenate((movement, messages))
        return action

    def apply_hunger_effects(self, messages):
        """Apply effects of hunger in agent messaging

        Args:
            messages (dict): Dictionary of recipient_agent_name: vector_message
        Returns:
            Modified message, according to hunger configuration
        """

        if self.hunger_config is None or self.hunger == 0:
            modified_messages = messages
        else:
            modified_messages = {}
            modify = lambda x: x
            effect = self.hunger_config["type"]
            hunger_ratio = self.hunger / MAX_HUNGER
            if effect == "noise_uniform":
                # add /subtract random noise up to a magnitude determined by hunger ratio
                params = self.hunger_config.get("params", [-hunger_ratio, hunger_ratio])
                modify = lambda x: x + np.random.uniform(*params, x.shape)
            elif effect == "noise_gaussian":
                # add random noise within a min / max window determined by hunger ratio
                params = self.hunger_config.get("params", [0, hunger_ratio])
                modify = lambda x: x + np.random.normal(*params, x.shape)
            # apply 
            modified_messages = modify(messages).clip(0, 1)
            new, old = [np.argmax(x) for x in (modified_messages, messages)]
            if new != old:
                log_store(self.communication_log, f"Changed message argmax from {old} to {new} due to hunger {self.hunger}", logging.debug)
        return modified_messages



    def reset(self):
        """Reset agent to an initial state
        """
        self.hunger = 0

    def get_attributes(self):
        return {
            "name": self.name,
            "hunger": self.hunger,
            "policy": self.policy_name,
            "weights": self.policy.get_weights()
            }