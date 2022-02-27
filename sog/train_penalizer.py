"""Penalizer training runnable functionality
"""

# native
from argparse import ArgumentParser
import logging
import pickle
import datetime

from os.path import join

# local
from sog.penalization.penalizers import instantiate
from sog.reporting import instantiate as instantiate_tracker
from sog.betrayal import BetrayalDatasetBuilder
from sog.utils import timestamp, configure_logging, log_to_stdout



parser = ArgumentParser()
parser.add_argument(
    '--penalizer', help='Penalizer class to discourage betrayal.',
    default='binary_mlp', type=str)
parser.add_argument(
    '--data_path', help='Training data path.', default=None, type=str)
parser.add_argument('--tracker', help='Experiment tracker.',
                    default=None, type=str)
parser.add_argument(
    '--run_path', help='Path to store run results.', default=f"penalizer_training_{timestamp()}", type=str)

parser.add_argument(
    '--run_name',
    help='Specify run name, allowing run reinitialization (useful for testing with a single run).',
    default=None)

args = parser.parse_args()

configure_logging(args.run_path)
log_to_stdout()

# tracking
exp_tracker = instantiate_tracker(which=args.tracker, run_name=args.run_name)
exp_tracker.update_config(args)

# make penalizer and load training data
penalizer = instantiate(args.penalizer)
penalizer.tracker = exp_tracker
with open(args.data_path, 'rb') as f:
    data = pickle.load(f)

# train
data, labels = BetrayalDatasetBuilder.preprocess_dataset(
    data, agent_name='learner', target="intended")
logging.info("Training penalizer")
penalizer.train(data, labels)

# store
model_path = join(args.run_path, penalizer.name + f"_{timestamp()}.pkl")

with open(model_path, 'wb') as f:
    logging.info(f"Saving penalizer model to {model_path}")
    pickle.dump({"config": vars(args), "models": penalizer.models, "results": penalizer.results, "name": penalizer.name, "data": {"path": args.data_path, **penalizer.data}}, f)
