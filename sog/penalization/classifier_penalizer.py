"""Module for classification-based penalization functionality
"""
# native
from functools import lru_cache
import logging
import json
from statistics import mean, stdev

# local
from sog.penalization.penalization import Penalizer

# third-party
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


class BinaryClassificationPenalizer(Penalizer):
    """Binary Classification Penalization  (BCP)
    Uses the sklearn estimator api

    Args:
        Penalizer (penalizer.Penalizer): Abstract penalizer class
    """
    def load(self, data, model_key='best_model_fit'):
        """Load trained model"""
        self.model = data['models']["best_cv_fit"]
        self.scaler = data['data']['scaler']

    def get_model(self):
        """Model constructor
        """
        raise NotImplementedError("Requiested abstract binary classification pernalizer")

    def preprocess_data(self, data: pd.DataFrame, means=None, stdevs=None):
        """Data preprocessor

        Args:
            data (pd.DataFrame): Input dataframe data
        """
        logging.info("Preprocessing data")
        sc = StandardScaler()
        sc.fit(data)
        data_s = sc.transform(data)
        return data_s, sc

    def train(self, data:np.ndarray, labels:np.ndarray):
        """Training function for BCP

        Args:
            data (np.ndarray): Training features
            labels (np.ndarray): Training labels
        """
        data_scaled, scaler = self.preprocess_data(data)
        self.data = {"inputs": data, "scaler": scaler}
        data = data_scaled
        logging.info("Training CV.")
        metric = "f1_macro"
        # scores = cross_validate(self.model, X=data, y=labels, scoring=metric, return_train_score=True)
        gscv = GridSearchCV(self.get_model(), cv=3, param_grid=self.get_cv_params(), scoring=metric, return_train_score=True)
        gscv.fit(data, labels)
        results, best_scores, estim, params = gscv.cv_results_, gscv.best_score_, gscv.best_estimator_, gscv.best_params_
        idx = gscv.best_index_

        self.results = {"best": {k: results[k][idx] for k in "params mean_train_score mean_test_score std_train_score std_test_score".split()}}
        self.results["cv"] = results

        retrained = self.get_model(**params).fit(data, labels)
        self.models = {"best_cv_fit": estim, "best_params": retrained}

        # baseline
        clf = DummyClassifier(strategy='stratified').fit(data, labels)
        baseline_ = cross_validate(clf, data, labels, return_train_score=True, scoring=metric)
        baseline = {f"baseline_mean_{k}": mean(baseline_[k]) for k in "train_score test_score".split()}
        baseline.update({f"baseline_std{k}": stdev(baseline_[k]) for k in "train_score test_score".split()})
        self.results["baseline"] = baseline

        self.tracker.log(self.results)

        logging.info("Baseline performance summary:")
        logging.info(json.dumps(baseline, indent=2))
        logging.info("Penalizer performance summary:")
        logging.info(json.dumps(self.results['best'], indent=2))

    def predict_penalization(self, instance, threshold=0.0):
        """Penalization prediction function

        Args:
            instance (pd.DataFrame): Input data with betrayal features from an agent action
            threshold (float): Prediction threshold to decide betrayal per message
        Returns:
            magnitude (float): Number in [0, 1] indicating whether the action was fully honest or fully betraying
        """
        instance_sc = self.scaler.transform(instance)
        if isinstance(instance, pd.DataFrame):
            instance = instance.values
        output = self.model.predict_proba(instance)
        # assume 2d output of class confidence for a) honest and b) betraying
        scores = output.squeeze()[1]
        # apply threshold and sum residuals
        magnitude = scores[scores>=threshold]
        return magnitude.mean()

class MLPPenalizer(BinaryClassificationPenalizer):
    """MLP classifier
    """
    name = "binary_mlp"
    def get_model(self, **kwargs):
        kwargs['hidden_layer_sizes'] = kwargs.get('hidden_layer_sizes', [200, 200])
        return MLPClassifier(**kwargs)
    @staticmethod
    def get_cv_params():
        return {
            "hidden_layer_sizes": [(100,), (200, 200), (300, 300, 300)],
            "max_iter": [100, 200, 500, 1000],
            "early_stopping": [True],
            "n_iter_no_change": [20],
            "solver": ["adam", "sgd"]
        }

        # # dummy
        # return { 
        #     "hidden_layer_sizes": [(10,), (10, 50,)],
        #     "max_iter": [10, 20],
        #     "early_stopping": [True],
        #     "n_iter_no_change": [5],
        #     "solver": ["adam", "sgd"]
        # }
