"""Utility functions
"""

# native
import json
import logging
import datetime

from os.path import exists, join
from os import makedirs
from pstats import Stats, SortKey
from contextlib import contextmanager
from cProfile import Profile

# third-party
import numpy as np

@contextmanager
def nop_wrapper(**kwargs):
    yield

@contextmanager
def profiling_wrapper(output_folder=""):
    """Function context for profiling.

    Args:
        output_folder (str, optional): Folder to write profiling information to. Defaults to "".
    """
    prof = Profile()
    prof.enable()
    yield
    # write profiling results
    prof.disable()
    output_path = join(output_folder, 'profiling_results.prof')
    with open(output_path, 'w') as fp:
        stats = Stats(prof, stream=fp)
        stats.strip_dirs().sort_stats('time').print_stats()
    stats.dump_stats(output_path + ".dump")
    logging.info(f"Profiling information at {output_path}")

def configure_logging(run_path, level=None, fmt='%(asctime)s,%(msecs)d %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s'):
    if not exists(run_path):
        makedirs(run_path)

    logfile = join(run_path, "logfile.log")
    print(f"Logging run to: {logfile}")
    level = level.upper() if isinstance(level, str) else level
    logging.basicConfig(level=level if level is not None else logging.DEBUG, filename=logfile, format=fmt)

def log_to_stdout():
    # add stdout logging
    fmt = logging.getLogger().handlers[0].formatter._fmt
    hdl = logging.StreamHandler()
    hdl.setFormatter(logging.Formatter(fmt=fmt))
    logging.getLogger().addHandler(hdl)


def timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def log_store(container, message, fn):
    fn(message)
    container.append(message)

def read_json(input_object: str):
    """Parse or load path from input json string

    Args:
        input_object (str):_Json object disk path or string dump

    Returns:
        The json object
    """
    if isinstance(input_object, dict):
        return input_object
    if exists(input_object):
        with open(input_object) as f:
            return json.load(f)
    return json.loads(input_object)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
