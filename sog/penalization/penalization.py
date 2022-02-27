"""Module for penalization functionality
"""

# third-party
import numpy as np

class Penalizer:
    """Abstract penalizer class
    """
    name = "PENALIZER"
    def __init__(self):
        """Constructor
        """
        self.model = self.get_model()

    def load(self, data):
        """Model loading function
        """
        raise NotImplementedError("Attempted to invoke abstract penalizer model loader")

    def get_model(self, **kwargs):
        """Model construction function
        """
        raise NotImplementedError("Attempted to invoke abstract penalizer model getter")

    def train(self, data: np.ndarray, labels: np.ndarray):
        """Train penalizer from input data and ground truth

        Args:
            data (np.ndarray): Training data
            labels (np.ndarray): Ground truth
        """
        raise NotImplementedError("Attempted to invoke abstract penalizer training")

    def predict_penalization(self, instance: np.ndarray) -> float:
        """Produce penalization score give input data

        Args:
            input_data (np.ndarray): _description_
        """
        raise NotImplementedError("Attempted to invoke abstract penalizer prediction")
