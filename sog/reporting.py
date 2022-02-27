"""Module for experiment tracking / reporting
"""
import logging
import json

import wandb
from wandb.integration.sb3 import WandbCallback

from sog.utils import timestamp, NumpyEncoder

def instantiate(which="wandb", run_name=None):
    """Instantiate experiment tracker

    Args:
        which (str, optional): Tracker name. Defaults to "wandb".
    """
    if which == "wandb":
        return WandbTracker(run_name=run_name)
    if which == "none" or which is None:
        return Tracker()
    raise NotImplementedError(f"Undefined tracker {which}")

class Tracker:
    """Abstraction for experiment tracking / reporting
    """
    def __init__(self):
        self.config = {}
    def log(self, key_value: dict):
        logging.info(json.dumps(key_value, indent=2, cls=NumpyEncoder))
    def update_config(self, config: dict):
        self.config.update(config)
    def conclude(self):
        pass
    def get_callback(self):
        return None
    def get_run_id(self):
        return timestamp()
    def set_run_name(self, name):
        self.name = name
    def get_lambda(cls):
        return lambda: Tracker()

class WandbTracker(Tracker):
    """Weights and biases
    """
    def get_lambda(self):
        return lambda: WandbTracker(self.run_name, self.group_name, run_obj=self.run)

    def __init__(self, run_name='run_name', group_name=None, run_obj=None):
        if run_obj is None:
            self.args = dict(project="sog", name=run_name, entity="sog", group=group_name, sync_tensorboard=True)
            self.run = wandb.init(**self.args)
        else:
            self.run = run_obj
            self.args = dict(project="sog", name=run_name, entity="sog", group=group_name, sync_tensorboard=True)
        self.run_id = self.run.id
        self.group_name = run_name or group_name or f"{self.run_name}_{self.run_id}"
        self.run_name = run_name

    def set_run_name(self, name):
        self.run.name = name

    def update_config(self, config: dict):
        wandb.config.update(config)

    def log(self, key_value: dict):
        wandb.log(key_value)

    def conclude(self):
        self.log({"completed": True})
        self.run.finish()
        print("Run at:", self.run.get_url())

    def get_callback(self):
        return WandbCallback()

    def get_run_id(self):
        return self.run.id
