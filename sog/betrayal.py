"""Module for betrayal-related logic and mechanics
"""

# native
import datetime
import logging
import pickle
from os.path import join
from collections import Counter

# third-party
import numpy as np
import pandas as pd

# local
from sog.agent import Policies
from sog.utils import timestamp


class BetrayalDatasetBuilder:
    """Class for betrayal dataset construction
    """
    label_types = ("intended", "honest", "accidental")

    def __init__(self):
        self.container = []

    def get_dataset(self):
        """Dataset getter function
        """
        return self.container

    def construct_betrayal_dataset_instance(
            self, agent_info, true_locations, intended_message, sent_message,
            label_info, hungers_dict):
        """Construct an instance of the betrayal-oriented dataset

        Args:
            container (iterable): Dataset container
            agent : Agent information to include (e.g. current policy weights, agent attributes, etc.)
            true_locations: True locations observed by the agent
            sent_message: Actual message sent
            label_info (int): Instance label
            hungers_dict (dict): Hunger values per agent
        """

        def policy_to_int(x):
            return list(Policies).index(Policies(x))

        instance = {**agent_info, **label_info}
        instance["policy"] = policy_to_int(instance["policy"])
        instance["true_loc"] = true_locations
        instance["intended_msg"] = intended_message
        instance["sent_msg"] = sent_message
        instance.update(hungers_dict)
        self.container.append(instance)

        return instance

    def flush_container_to_disk(self, output_folder: str=".", add_timestamp=False):
        """Write accumulated dataset to disk and empty container
        """

        logging.info("Dataset stats:")
        for label in BetrayalDatasetBuilder.label_types:
            logging.info(f"{label} true count: {len([x for x in self.container if x[label]])}")
        
        output_path = join(output_folder, f"penalization_dataset{'_' + timestamp() if add_timestamp else ''}.pkl")

        logging.info(f"Writing {len(self.container)}-sized betrayal dataset to {output_path}")

        with open(output_path, "ab") as f:
            pickle.dump(self.container, f)
        self.container = []

    def calculate_location_alignment(self, locs, msg):
        """Compares a vector of existing food locations with an agent message, to determine alignment between them

        Args:
            locs (np.ndarray): True food locations
            msg (np.ndarray): Outgoing message
        """
        # steer comparison towards the most prominent location the agent picked
        # to communicate
        # things to note:
        # a) this is guaranteed to work for an honest receiver, that uses argmax to deduce location
        # b)this should converge to working for an honest sender and a generic receiver,
        # since the former uses one-hot encoding of a food location indication
        # for a generic vector encoding / decoding scheme, however, we would need a different approach
        picked_location = np.argmax(msg)
        return (not locs.any()) or locs[picked_location] == 1

    def analyze_iteration(self, food_locations, agent_info, messages, intended_messages, agents_dict, agent_histories):
        """Analyze an iteration to deduce whether betrayal occured. If so, derive a betrayal dataset instance

        Args:
            food_locations (np.ndarray): Binary food locations vector
            agent_info (dict): Agent atributes name-value dict
            messages (dict): Transmitted agent messages
            intended_messages (dict): Intended agent messages (without, e.g. hunger effects)
        """

        sender_name = agent_info['name']
        sender = agents_dict[sender_name]
        instances, label_infos = [], []
        for recipient_name, imsg in intended_messages.items():
            locs, smsg = food_locations[recipient_name], messages[recipient_name]
            label_info = self.calculate_ground_truth(locs, imsg, smsg, agent_info)
            label_info['sender'] = sender_name
            label_info['recipient'] = recipient_name
            logging.debug(f"Betrayal label info: {label_info}")

            recipient = agents_dict[recipient_name]
            metadata = {
                "hunger_sender": sender.get_hunger(), "hunger_receiver": recipient.get_hunger(), **agent_histories[sender_name], **agent_histories[recipient_name]
                }

            instance = self.construct_betrayal_dataset_instance(
                agent_info, locs, imsg, smsg, label_info, metadata)
            instances.append(instance)
            label_infos.append(label_info)
        return instances, label_infos
        # else:
        #     raise NotImplementedError(
        #         "Need to create betrayal dataset generation for policy:" + agent_info["policy"])

    def calculate_ground_truth(self, true_locations, intended_message, sent_message, agent_info):
        """Calculate instance ground truth

        Args:
            true_locations (np.ndarray): True food lcoations
            intended_message (np.ndarray): Food location the agent intended to send
            sent_message (np.ndarray): Food locations the agent indeed sent
            agent_info (np.ndarray): Additonal agent info

        Returns:
            Label ground truth information
        """
        # willing vs. accidental (hunger-induced) deception
        intended_betrayal = not self.calculate_location_alignment(true_locations, intended_message)
        accidental_betrayal = not self.calculate_location_alignment(true_locations, sent_message) and not intended_betrayal

        label_info = {
            "intended": intended_betrayal,
            "accidental": accidental_betrayal,
            "honest": not(intended_betrayal or accidental_betrayal)}
        return label_info

    @staticmethod
    def preprocess_dataset(data, agent_name, target="intended"):
        """Perform baseline featurization / preprocessing on the input betrayal dataset

        Args:
            data (list): Filtered / preprocessed dataset
        """
        # to dataframe
        full_df = pd.DataFrame.from_records(data)
        # only use the learner's data
        full_df = full_df[full_df['name'] == agent_name]
        # use features as-is: hungers, policies
        as_is, arrays = [], []
        as_is += ["hunger_sender", "hunger_receiver"]
        # all cumulative scores
        # as_is += [x for x in full_df.columns if x.startswith("cumulative_")]

        # arrays: sent message and model weights
        # true locations will be unknown during deployment and will not be used for training
        # arrays += ["sent_msg"]
        arrays += ["sent_msg", "true_loc"]
        # arrays += ["sent_msg", "weights"]
        # arrays += ["sent_msg", "true_loc", "weights"]
        # all collected history data
        # arrays += [x for x in full_df.columns if x.startswith("history_")]

        # build training df
        df = pd.DataFrame.from_records(full_df[as_is])
        for k in df.columns:
            assert len(df[k].shape) == 1 or  df[k].shape[-1] == 1, f"Unexpected scalar shape for {k}: {df[k].shape} -- {df[k]}"
        assert not np.isnan(df.values).any(), "Nan values in betrayal data processing"
        logging.info(f"Using scalar data: {as_is}")
        for arr in arrays:
            values = np.vstack(full_df[arr].values)
            logging.info(f"Using array data: {arr} of dim {values.shape[-1]}")
            a_df = pd.DataFrame(columns=[f"{arr}_{i}" for i in range(values.shape[-1])], data=values)
            df = pd.concat((df, a_df), axis=1)
        if target in full_df:
            labels = full_df[target].values.astype(np.int32)
        else:
            labels = None
        logging.info(f"Final training data dims: {df.shape}")
        logging.info(f"Final label distro data dims: {Counter(labels).most_common()}")
        if np.isnan(df.values).any():
            print()
        return df, labels