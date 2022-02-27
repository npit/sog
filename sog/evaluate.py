"""Module for run evaluations
"""
# native
import logging
import json
from copy import deepcopy
from collections import defaultdict
from statistics import mean

# third-parte
import pandas as pd
import numpy as np

# local
from sog.utils import timestamp
from sog.betrayal import BetrayalDatasetBuilder

class Evaluator:
    """Class for evaluating SSR runs
    """
    def __init__(self, max_iterations, num_episodes, tracker_lambda=None):
        """Constructor

        Args:
            tracker (tracker.Tracker, optional): Experiment tracker object. Defaults to None.
        """
        self.max_iterations = max_iterations
        self.num_episodes = num_episodes

        self.labels = defaultdict(list)
        self.episode = self.iteration = 0
        self.agents = set()
        self.history = defaultdict(list)

        self.tracker_lambda = tracker_lambda
        self.tracker = None

    def initialize(self):
        """Prepare for evaluation and tracking
        """
        self.tracker = self.tracker_lambda()

    def set_iteration(self, iteration):
        self.iteration = iteration

    def set_episode(self, episode):
        self.episode = episode

    def save_history(self, path=""):
        path = path or f"history_{timestamp()}"
        with open(path, 'w') as f:
            json.dump(self.history, f)

    def process_iteration(self, action, env, reward_delta, agent, nutrition_received, betrayal_info, betrayal_estimate=None):
        """Process results from a single iteration for evaluation

        Args:
            observation: Iteration observation
            reward : Iteration neward
            env: Environment
        """
        if self.tracker is None:
            self.initialize()

        self.agents.add(agent)
        # data to track
        data = {
            f"{agent.name}_reward": reward_delta,
            f"{agent.name}_hunger": agent.hunger,
            "iteration": self.iteration,
            "agent": agent.name,
            f"{agent.name}_nutrition": nutrition_received,
            "episode": self.episode
            }
        # betrayal logging
        # -----------------
        if betrayal_info is not None:
            assert len(betrayal_info) == 1, f"Multiagent setting unsupported"
            recipient = betrayal_info[0]['recipient']
            # automatic betrayal estimates
            if betrayal_estimate is not None:
                data[f'{agent.name}_betrayal_estimate_to_{recipient}'] = betrayal_estimate
            # betrayal values -- iterate individual messages.
            betrayal_info = betrayal_info or []
            betrayal_averages = defaultdict(list)
            # group per-recipient betrayal scores
            for x in betrayal_info:
                recipient = x['recipient']
                # iterate individual labels
                for k, v in x.items():
                    if k in "sender recipient".split():
                        continue
                    betrayal_averages[k].append(int(v))
                    data[f"{agent.name}_{k}_to_{recipient}"] = int(v)

            # group average betrayal scores across all recipients
            for k, v in betrayal_averages.items():
                data[f"{agent.name}_{k}_toall_mean"] = mean(v)

        # flat_labels = defaultdict(list)
        # for x in betrayal_info:
        #     for k, v in x.items():
        #         flat_labels[k].append(int(v))

        # for btype, values in flat_labels.items():
        #     self.labels[btype].extend(values)
        # for k, v in flat_labels.items():
        #     data[f"cumulative_{k}"] = sum(v)

        logging.debug(f"Iteration {self.iteration+1}/{self.max_iterations}  -- [{action}], reward: {reward_delta}")
        cumul_data = self.make_cumulative_quantities(data)
        data.update(cumul_data)
        self.history['iterations'].append(data)
        self.tracker.log(data)


    def make_cumulative_quantities(self, data):
        data = {k: v for (k, v) in data.items() if not (isinstance(v, str) or k in "_timestamp _runtime episode iteration".split())}
        current = {}
        for k, v in data.items():
            ck = "cumulative_" + k
            value = data[k]
            try:
                prev = self.history[ck][-1]
                value += prev
            except (KeyError, IndexError):
                pass
            self.history[ck].append(value)
            current[ck] = value
        return current

    def get_agent_history(self, agent_name, size=5):
        # get cumulative scores
        results = {f"cumulative_{agent_name}_{quantity}": 0 for quantity in "reward hunger".split()}
        results.update({f"cumulative_{agent_name}_{quantity}_mean": 0 for quantity in ("intended_toall".split())})
        for k in results:
            if k not in self.history:
                continue
            results[k] = self.history[k][-1]

        # get local history
        results_series = {f"{agent_name}_{quantity}": np.zeros((size,), dtype=np.float32) for quantity in ("reward hunger".split())}
        for k in results_series:
            if 'iterations' not in self.history:
                continue
            data = [x[k] for x in self.history['iterations'] if k in x][-size:]
            if data:
                results_series[k][-len(data):] = data
            else:
                print()
        # rename history keys
        keys = list(results_series.keys())
        for k in keys:
            results_series[f"history_{k}"] = results_series[k]
            del results_series[k]
        return {**results, **results_series}

    def post_process_episode(self):
        """Postprocess data collected from all episode iterations
        """
        data = {"episode": self.episode}
        for agent in self.agents:
            relevant_history = [h for h in self.history['iterations'] if h['episode'] == self.episode and h['agent'] == agent.name]
            dat = self.process_collection(relevant_history, agent, aggregation_name="episode")
            data.update(dat)
        self.tracker.log(data)
        self.history['episodes'].append(data)

    def process_collection(self, collection, agent, aggregation_name):
        """Process and aggregate iteration data collection

        Args:
            collection (_type_): _description_
            agent (_type_): _description_
            aggregation_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = {}
        df = pd.DataFrame.from_records(collection)

        # betrayal stats
        for label in BetrayalDatasetBuilder.label_types:
            if not any(label in c for c in df.columns):
                continue
            for recipient in self.agents:
                if recipient == agent:
                    continue
                data[f"{agent.name}_to_{recipient.name}_{aggregation_name}_{label}_ratio"] = df[f"{agent.name}_{label}_to_{recipient.name}"].mean()
        
            # overall average
            data[f"{agent.name}_toall_{aggregation_name}_{label}_ratio"] = df[f"{agent.name}_{label}_toall_mean"].mean()

        for quantity in "reward nutrition hunger".split():
            data[f'{agent.name}_{aggregation_name}_mean_{quantity}'] = df[f"{agent.name}_{quantity}"].mean()
        return data


    def post_process_evaluation(self):
        """Produce overall aggregation
        """
        data = {}
        for agent in self.agents:
            relevant_history = [h for h in self.history['iterations'] if h['agent'] == agent.name]
            dat = self.process_collection(relevant_history, agent, aggregation_name="total")
            data.update(dat)

        self.tracker.log(data)
        self.history['total'] = data
