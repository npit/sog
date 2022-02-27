"""Penalizers instantiation fucntionality
"""
# native
from os.path import exists
import pickle

# local
from sog.penalization.classifier_penalizer import *

available_penalizers = [MLPPenalizer]

def instantiate(name_or_path):
    """Penalizer instantiation function

    Args:
        name (str): Penalizer name

    Raises:
        NotImplementedError: Upon undefined penalizer request

    Returns:
        Penalizer: The instantiated penalizer object
    """
    data = None
    if exists(name_or_path):
        with open(name_or_path, "rb") as f:
            data = pickle.load(f)
        name = data["name"]
    else:
        name = name_or_path

    penalizer_object = None
    for penalizer_class in available_penalizers:

        if penalizer_class.name == name:
            penalizer_object = penalizer_class()
            if data:
                penalizer_object.load(data)
            return penalizer_object
    raise NotImplementedError(f"Undefined penalizer: {name}, available are: {[x.name for x in available_penalizers]}")