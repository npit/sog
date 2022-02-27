import random
from collections import defaultdict
import logging
import json

# third
import numpy as np
from gym.spaces import Box
from gym import Env

# project
from sog.world import World
from sog.agent import Agent
from sog.food import Food, Nutrition
from sog.presets import available_presets
from sog.utils import read_json


class SOG(Env):
    """Symmetric observer-gatherer environment class
    """

    metadata = {
        "render.modes": ["human"],
        "name": "symmetric sender-receiver",
        "is_parallelizable": False,
        "video.frames_per_second": 1,
    }

    @staticmethod
    def get_agent_init_func(preset: dict, run_path: str=".", policy_load_config: dict=None):
        """Obtain agent initialization function

        Args:
            preset (dict): Agent preset setting to utilize
            run_path (str, optional): Experiment run path. Defaults to ".".
            policy_load_config (dict, optional): Policy loading configuration. Defaults to None.

        Returns:
            _type_: _description_
        """
        if isinstance(preset, str):
            preset = available_presets[preset]
        policy_init_funcs = {}
        policy_load_config = policy_load_config or {}
        agent_names = [x['name'] for x in preset]
        for agent_name in agent_names:
            def fn(env, name=agent_name):
                policy = env.agents_dict[name].get_policy()
                policy.initialize(env, prefix=str(env._instance_id) + "_" + name, run_path=run_path, policy_path=policy_load_config.get(name))
            policy_init_funcs[agent_name] = fn
        return policy_init_funcs


    def get_configuration(self):
        """Configuration getter"""
        return {
            "presets": self.agent_presets,
            "nutrition_config": self.nutrition_config,
            "world_sizes": self.world_sizes
        }

    def __init__(self, agent_presets=None, world_size=5, message_dim=None, agents_start_location="midpoint", max_food_per_world=None,
                nutrition_config="{}", no_eat_penalty=0, seed=42,
                hunger_delta=1, penalizer=None, learning_agent_name='learner', acting_order="static", round_begins_with_messaging=False, observe_hunger=False, penalized_agents="learner", penalization_magnitude=None, apply_penalization=False,
                agent_initializer_funcs=None, instance_id=None, betrayal_analyzer=None, evaluator=None):
        """Constructor

        Args:
            world_sizes (int): World size per agent.
            message_dim (int, optional): Dimension for message vector. Defaults to None, corresponding to the world size.
            agents_start_location (str): Where to spawn an agent in the world
            max_food_per_world (int): Maximum number of food items to spawn per world. Defaults to nearest int of 2 x world_size / 3.
            nutrition_config (dict): Nutrition configuration
            no_eat_penalty (int): Reward penalty for each agent action without food consumption. Defaults to 0.
            seed (int): Random seed. Defaults to 42.
            hunger_delta (int): Hunger increase per turn. Defaults to 1.
            penalizer (Penalizer): Penalization object.
            learning_agent_name (str): Name for the learning agent.
            agent_presets (dict, optional): Dict specifying agent behaviour
            acting_order (str, optional): Acting order of agents. Can be "static" or "dynamic"
            round_begins_with_messaging (bool, optional): If true, each round begins by gathering agent messages, such that the first acting agent has messages available.
            observe_hunger (bool): Whether an agent can directly observe another's hunger level
            penalized_agents (str): Comma-separated agent names to penalize
            penalization_magnitude (float): A positive number specifying the maximum penalization value. If None, defaults to max nutrition.
            apply_penalization (boolean): Whether to apply the computed penalization in the agent reward
            agent_initializer_funcs (lambda): Functions to initialize each agent
            penalizer (Penalizer): The penalizer object to use
            betrayal_analyzer (BetrayalAnalyzer): Betrayal analysis object
            instance_id (int): Environment identifier
        """
        super().__init__()

        # for debugging multiprocessing
        self._instance_id = instance_id

        # assignments for SB3 PPO compatibility
        max_penalizer_penalization = 0 if penalizer is None else penalizer.get_max_penalization()
        self.reward_range = (no_eat_penalty + max_penalizer_penalization, Nutrition.get_max())

        if agent_presets is None:
            agent_presets = available_presets['ppo_honest']
        for i in range(len(agent_presets)):
            agent_presets[i]['index'] = i

        # logistical stuff
        assert max_food_per_world is None or max_food_per_world >= 1, "Max food per world has to be >= 1"

        self.observe_hunger = observe_hunger
        self.nutrition_config = read_json(nutrition_config)
        self.max_food_per_world = max_food_per_world
        self.no_eat_penalty = no_eat_penalty
        self.agent_presets = agent_presets
        self.iteration = 0
        self.round = 0
        self.hunger_delta = hunger_delta
        self.acting_order = acting_order
        self.round_begins_with_messaging = round_begins_with_messaging
        self.betrayal_analyzer = betrayal_analyzer
        self.evaluator = evaluator
        self.penalizer = penalizer
        self.penalized_agents = penalized_agents.split(",")
        self.penalization_magnitude = Nutrition.get_max().value if penalization_magnitude is None else penalization_magnitude
        self.apply_penalization = apply_penalization

        # build world containers
        world_sizes = [(world_size,) for _ in agent_presets]
        self.worlds = [World(s, size) for (s, size) in enumerate(world_sizes)]
        world_sizes = [world.size_1d for world in self.worlds]
        self.world_sizes = world_sizes

        self.create_food()

        # configure agents
        self.agents_start_location = agents_start_location
        self.agent_objects = [ Agent(**preset) for preset in agent_presets]

        self.agent_names = [a.name for a in self.agent_objects]
        self.agents = self.possible_agents = self.agent_names

        self.agents_dict = {x.name: x for x in self.agent_objects}

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        # we need a) an action space for the transmitted communication vector
        # we need b) an action space for the grid location to move to
        # communication: N one-hot vectors for a world of size N

        # messages pool
        self.messages_from_to = defaultdict(dict)

        self.observation_spaces = {}
        self.action_spaces = {}

        if message_dim is None:
            # an agent sents a message of eq. size to the observed world
            message_dims = world_sizes
        elif isinstance(message_dim, int):
            # constant vector dim across agents and worlds
            self.agent_message_dims = [message_dim for _ in world_sizes]
        else:
            # a message dim per agent and world
            # TODO
            raise NotImplementedError(
                "Only fixed and per-world message dim for now")

        self.agent_message_dims = message_dims

        self.reward = 0
        self.done = False
        # self.infos = {i: {'legal_moves': list(
        #     range(0, 9))} for i in self.agents}

        self.learning_agent = self.agents_dict[learning_agent_name]
        if acting_order == "static":
            self.agent_acting_order = random.sample(self.agent_names, len(self.agent_names))

        if self.penalizer:
            invalid_agents = (x in self.agents_dict for x in self.penalized_agents)
            assert not any(invalid_agents), f"Undefined agents to penalize: {invalid_agents}"

        self.observation_space = self.make_flat_obs_space()
        self.action_space = self.make_flat_action_space()
        if agent_initializer_funcs is None:
            for agent in self.agent_objects:
                agent.get_policy().initialize(self)
        else:
            for agent_name, fn in agent_initializer_funcs.items():
                fn(self)

    def update_learner_model(self, model_weights):
        """Function to update the policy model of the learning agent.
        Useful for syncing the model in multiproc vectorized envs.

        Args:
            model: The policy model object
        """
        logging.debug(f"Env w/ seed: {self.seed} updating learner model.")
        self.learning_agent.policy.update_model(model_weights)

    def make_flat_action_space(self):
        """Function to create a flat action space, rather than nested dict
        Returns:
            Box: The action space
        """
        # concatenated message to other agents
        viewed_worlds = self.get_observed_worlds(self.learning_agent.name)
        size_sum = sum(w.size_1d for w in viewed_worlds)
        # movement
        movement_dim = self.get_world_of_agent(self.learning_agent.name).size_1d
        # return a single box
        return Box(low=0.0, high=1.0, shape=(size_sum + movement_dim, ))

    def get_incoming_messages(self, recipient, concatenate=False):
        """ Retrieve all cached messages for the input agent
        """
        senders = self.get_other_agents(recipient)
        msg_dim = self.get_world_of_agent(recipient).size_1d
        messages = []
        for sender in senders:
            try:
                msg = self.messages_from_to[sender][recipient]
                messages.append(msg)
            except KeyError:
                messages.append(np.zeros((msg_dim,)))
        if concatenate:
            messages = np.concatenate(messages, axis=0)
        return messages

    def get_other_agents(self, agent_name: str):
        if isinstance(agent_name, Agent):
            agent_name = agent_name.name
        elif isinstance(agent_name, str):
            pass
        else:
            raise ValueError(f"Invalid agent input: {agent_name}")
        return [x for x in self.agent_names if x != agent_name]

    def make_flat_obs_space(self):
        """Function to create a flat observation space, rather than nested dict
        Returns:
            Dict: The observation space
        """
        obs_space = None
        for agent in self.agent_objects:
            # concatenated observation of other worlds
            viewed_worlds = self.get_observed_worlds(agent.name)
            observed_sizes = [w.size_1d for w in viewed_worlds]
            size_sum = sum(observed_sizes)
            # worlds = Box(low=0.0, high=1.0, shape=(size_sum,))
            # concatenated observations of incoming messages
            message_dim = self.get_world_of_agent(agent.name).size_1d
            all_messages_dim = sum(message_dim for _ in viewed_worlds)

            # assign dimensions for obs decoding to the learning agent
            agent.set_world_size(message_dim)
            agent.set_observed_world_sizes(observed_sizes)

            
            if agent == self.learning_agent:
                n_hungers = 1 + (len(self.get_other_agents(agent)) if self.observe_hunger else 0)
                # conatenate all to a single box
                obs_space = Box(low=0.0, high=1.0, shape=(size_sum + all_messages_dim + n_hungers,))
            
        return obs_space


    def create_food(self):
        """Food instantiation function. Food items are created in a flat pool and stored
        """
        # create food items
        # max food at 2/3 of world size, if not defined
        max_food_per_world = [random.randint(1, self.max_food_per_world or max(1, 2* w.size_1d //3)) for w in self.worlds]
        max_food_per_world = [np.clip(mx, 1, w.size_1d) for (mx, w) in zip(max_food_per_world, self.worlds)]
        num_total_food = sum(max_food_per_world)
        food = [Food() for _ in range(num_total_food)]
        return food


    def get_observed_worlds(self, agent_name):
        """Retrieved worlds observed by the agent specified by the input name

        Args:
            agent_name (str): Name of observing agent

        Returns:
            list: Worlds observed by the input agent
        """
        res = []
        for i, w in enumerate(self.worlds):
            if self.agents[i] != agent_name:
                res.append(w)
        return res
        # return [w for w in self.worlds if agent_name != w.agent.name]


    def get_world_of_agent(self, agent_name:str):
        """Retrieve world inhabited by input agent

        Args:
            agent_name (str): Name of the agent

        Returns:
            worldWorld: World object
        """
        for a, w in zip(self.agents, self.worlds):
            if a == agent_name:
                return w
        return None

    def get_observed_food_locations(self, agent_name):
        """Retrieved food locations observed by the agent specified by the input name

        Args:
            agent_name (str): Name of observing agent

        Returns:
            list: Food location vectors observed
        """
        result = {}
        for agent, world in zip(self.agent_objects, self.worlds):
            if agent.name == agent_name:
                continue
            result[agent.name] = world.get_food_one_hot()
        return result

    def reallocate_food(self, food_items=None):
        """Randomly reallocate food items in the gridworlds
        """
        if food_items is None:
            # gather food from the worlds
            food_items = self.get_available_food()

        for w in self.worlds:
            w.clear_food()

        # assign food amounts per world
        candidate_world_idx = list(range(len(self.worlds)))
        while food_items:
            food_item = food_items.pop()
            world_idx = candidate_world_idx.pop(0)
            world = self.worlds[world_idx]
            if not world.can_insert_food():
                continue
            # randomly place food in the world
            world.place_food(food_item)
            # FIFO
            candidate_world_idx.append(world_idx)

    def state(self):
        return self.states

    def observe(self, agent):
        """
        An agent may observe
        - The opposite world(s)
        - Messages from the other agents
        """
        return self.observation_space(agent)

    def compute_reward(self, food_consumed=None, compute_penalization=False, penalization_input=None):
        """Compute agent reward

        Args:
            message (_type_): _description_
            food_consumed (_type_, optional): _description_. Defaults to None.
            dishonesty_penalty (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # reward from nutrition
        rewards, penalties = [], []
        if food_consumed:
            rewards.append(food_consumed.get_reward_value())
        else:
            penalties.append(self.no_eat_penalty)

        estimated_betrayal = 0
        if compute_penalization:
            estimated_betrayal = self.penalizer.predict_penalization(penalization_input)
            if self.apply_penalization:
                # modify the penalization by th
                penalty = - estimated_betrayal * self.penalization_magnitude 
                logging.debug(f"Applying penalization {estimated_betrayal} * {self.penalization_magnitude} = {penalty}")
                penalties.append(penalty)
        reward_delta = sum(rewards + penalties)
        return reward_delta, estimated_betrayal

    
    def handle_agent_action(self, agent, action=None):

        if action is None:
            action = agent.apply_policy(self)
        # get agent info before they undergo any status change due to (not) eating
        agent_info = agent.get_attributes()
        self.render(input_agent=agent)

        # add a nutrition decay / hunger increase after the action
        agent.increase_hunger(self.hunger_delta)

        # agent movement
        movement, messages_vector = agent.parse_action(action)
        movement_int = int(np.argmax(movement))
        agent_world = self.get_world_of_agent(agent.name)
        food_consumed = agent_world.attempt_set_food_consumed(movement_int)
        if food_consumed:
            agent.consume_food(food_consumed)


        # betrayal
        penalization_input, betrayal_labels = None, None
        if self.betrayal_analyzer is not None:
            food_locations = self.get_observed_food_locations(agent.name)
            intended_messages_per_recipient = self.split_message_to_recipients(agent.name, agent.get_intended_messages())
            messages_per_recipient = self.split_message_to_recipients(agent.name, messages_vector)
            histories = {}
            for agent_name in self.agents_dict:
                histories[agent_name] = self.evaluator.get_agent_history(agent_name)

            betrayal_instances, betrayal_labels = self.betrayal_analyzer.analyze_iteration(food_locations, agent_info, messages_per_recipient, intended_messages_per_recipient, self.agents_dict, histories)

        compute_penalization = self.penalizer is not None and agent.name in self.penalized_agents
        if compute_penalization:
            penalization_input, _ = self.betrayal_analyzer.preprocess_dataset(betrayal_instances, agent.name)

        # reward
        reward_delta, betrayal_estimate = self.compute_reward(food_consumed, compute_penalization, penalization_input)

        # messaging
        ###########################################
        self.collect_messages(agent.name, messages_vector)


        rstep = 1 + self.iteration % len(self.agents)
        logging.info(f"Round: {self.round}, rstep {rstep} / {len(self.agents)}, step: {self.iteration}, {self.get_agent_descr(agent)}: moved to {movement_int:2d}, [{'HIT' if food_consumed is not None else 'MISS'}] got {reward_delta} reward from food: {food_consumed} -- hunger now {agent.hunger}.")
        self.rewards[agent.name] += reward_delta

        # post
        if self.evaluator is not None:
            nutrition_received = food_consumed.get_nutrition_value() if food_consumed else 0
            self.evaluator.process_iteration(action, self, reward_delta, agent, nutrition_received, betrayal_labels, betrayal_estimate)


        self.iteration += 1
        return reward_delta

    def get_available_food(self):
        """Collect available food from all worlds

        Returns:
            list: List of available food items
        """
        food_items = [f for w in self.worlds for f in w.foods.values()]
        return food_items


    def split_message_to_recipients(self, sender: str, composite_message: np.ndarray):
        """Split a vector of composite messages to each recipient

        Args:
            sender (str): Sender agent name
            composite_message (np.ndarray): The composite message

        Returns:
            dict: A recipient-message vector mapping
        """
        result = {}
        for recipient in self.get_other_agents(sender):
            dim = self.get_world_of_agent(recipient).size_1d
            result[recipient], composite_message = composite_message[:dim], composite_message[dim:]
        assert len(composite_message) == 0, "Message ingestion incomplete!"
        return result

    def collect_messages(self, sender: str, composite_message: np.ndarray):
        """Store messages received from an agent action

        Args:
            sender (str): Sender agent
            messages (dict): Concatenated messages vector
        """
        recipient_messages_dict = self.split_message_to_recipients(sender, composite_message)
        for recipient, msg in recipient_messages_dict.items():
            self.messages_from_to[sender][recipient] = msg

    def advance_environment(self, which):
        """Function to execute actions of background agents and update the environment accordingly

        Args:
            which (str, optional): Stage relative to learner agent.
        """
        if which == "pre" and self.round_begins_with_messaging:
            logging.debug(f"Gathering messages for round {self.round}")
            for agent in self.agent_objects:
                action = agent.apply_policy(self)
                _, messages_vector = agent.parse_action(action)
                self.collect_messages(agent.name, messages_vector)

        for i, name in enumerate(self.background_agents[which]):
            self.handle_agent_action(self.agents_dict[name])

    def get_agent_descr(self, agent):
        """Helper function for agent print information retrieval

        Args:
            agent (Agent): Agent object

        Returns:
            String description of useful agent information
        """
        for stage in self.background_agents:
            stage_agents = self.background_agents[stage]
            if agent in stage_agents:
                idx, num = stage_agents.index(agent), len(stage_agents)
                return f"[{stage} {idx+1}/{num}]"
        return '[learner]'


    def step(self, action):
        # do learning agent actions
        reward_delta = self.handle_agent_action(self.learning_agent, action)
        # agents after the learning agent 
        self.advance_environment("post")

        # configure a new round
        self.initialize_new_round()
        # do actions of agents acting before the learning agent
        self.advance_environment("pre")

        # we're done if no food remains across all worlds
        if not self.get_available_food():
            self.done = True

        ###########################################
        return self.get_state(self.learning_agent.name), reward_delta, self.done, self.infos

    def initialize_new_round(self):

        # decide acting order
        if self.acting_order == "dynamic":
            self.agent_acting_order = random.sample(self.agent_names, len(self.agent_names))

        # food relocation
        self.reallocate_food()

        self.round += 1
        learn_idx = self.agent_acting_order.index(self.learning_agent.name)
        self.background_agents = {"pre": self.agent_acting_order[:learn_idx], "post": self.agent_acting_order[learn_idx+1:]}
        logging.debug(f"Acting order: {self.acting_order} pre: {self.background_agents['pre']}, learner: {self.learning_agent.name},  post: {self.background_agents['post']}")

    def reset_agent_locations(self):
        """Reset location of each agent in their world
        """
        # randomize agent locations
        for w, world in enumerate(self.worlds):
            world.place_agent(self.agent_objects[w], self.agents_start_location)

    def reset(self):
        logging.debug("Reseting...")
        # reset environment
        self.agents = [x.name for x in self.agent_objects]
        [x.reset() for x in self.agent_objects]
        self.states = {agent: None for agent in self.agent_names}

        [world.reset() for world in self.worlds]
        self.reset_agent_locations()
        food = self.create_food()
        self.reallocate_food(food)

        self.messages_from_to = defaultdict(dict)

        self.rewards = {i: 0 for i in self.agent_names}
        self.done = False
        self.infos = {i: {} for i in self.agent_names}
        self.round = -1
        self.iteration = 0
        
        # decide agent order
        self.initialize_new_round()

        # do agent actions before the learner, if any
        self.advance_environment("pre")
        return self.get_state(self.learning_agent.name)

    def get_state(self, agent_name):
        """ Get state vector from an agent perspective """
        components = []
        messages = self.get_incoming_messages(agent_name, concatenate=True)
        worlds = self.get_world_observations(agent_name)
        components.extend([messages, worlds])
        # add hunger score of self
        hungers = [self.agents_dict[agent_name].get_hunger()]
        if self.observe_hunger:
            # hunger score of others
            hungers.extend([self.agents_dict[agent_name].get_hunger() for agent_name in self.get_other_agents(agent_name)])
        components.append(hungers)
        return np.concatenate(components, axis=0)

    def get_world_observations(self, agent_name):
        worlds = self.get_observed_worlds(agent_name)
        worlds_vector = np.concatenate([w.get_food_view() for w in worlds])
        return worlds_vector

    def render(self, mode='human', input_agent=None):
        if mode == 'human':
            for w, world in enumerate(self.worlds):
                if input_agent is not None:
                    if world.agent.name != input_agent.name:
                        continue
                    agent = input_agent
                else:
                    agent = world.agent
                wrld = world.render(np.argmax(agent.last_movement))
                agent_name = self.agents[w]
                logging.info(f"{wrld} {self.agents[w]}, nfood: {len(world.foods)} current reward: {self.rewards[agent_name]:.4f}")
        else:
            raise ValueError(f"Unsupported rendering mode: {mode}")
