"""
PLEASE REMEMBER TO UPDATE THE FILENAME AND PATH IN `aicrowd.yaml` AS NEEDED.
"""

import glob
import os
import signal
import tempfile
import time
from contextlib import contextmanager

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

from evaluation_utils.base_predictor import BaseCLEARPredictor
from evaluation_utils.metrics import *


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    '''whether model inference timed out
    Args:
        seconds: length of timeout
    '''
    def signal_handler(signum, frame):
        raise TimeoutException("Prediction timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class CLEAREvaluator:
    def __init__(
        self,
        test_data_path: str,
        models_path: str,
        predictions_file_path: str,
        save_file_path: str,
        predictor: BaseCLEARPredictor = None,
    ):
        '''
        Args: 
            test_data_path(str list): folder path of images data
            models_path(str list): folder path of 10 models
            predictions_file_path(str list): folder path of predictions files
            save_file_path(str list): folder path of save files
            prediction_setup_timeout(int): limit time for loading model
            prediction_timeout(int): limit time for inferencing
            predictor: predictor of inference
            predictions: result of inference
            save_name(str): name of plot image
            title(str): title of plot image
        '''
        self.test_data_path = test_data_path
        self.models_path = models_path
        self.predictions_file_path = predictions_file_path
        self.save_file_path = save_file_path

        self.prediction_setup_timeout = 120
        self.prediction_timeout = 60000
        self.predictor = predictor
        self.predictions = None
        self.save_name = "accuracy_matrix"
        self.title = "Accuracy Matrix"

    def validate_predictions(self, prediction):
        assert isinstance(prediction, np.ndarray)

    def save_predictions(self):
        '''save model inference result
        Args:
            predictions: result of inference
            predictions_file_path(str list): folder path of predictions files
        '''
        fd, fpath = tempfile.mkstemp(dir=os.path.dirname(self.predictions_file_path))
        np.savetxt(fpath, self.predictions)
        os.rename(fpath, self.predictions_file_path)

    def evaluation(self):
        ''' Inference function
        '''
        try:
            with time_limit(self.prediction_setup_timeout):
                self.predictor.prediction_setup(self.models_path)
        except NotImplementedError:
            pass

        with time_limit(self.prediction_timeout):
            prediction = self.predictor.prediction(self.test_data_path)

        self.validate_predictions(prediction)
        self.predictions = prediction
        self.save_predictions()

    def scoring(self, prediction_file_path, partial_score: bool = False):
        '''Plot accuracy matrix
        Args:
            predictions_file_path(str list): folder path of predictions files
            partial_score(bool): False
        '''
        predictions = np.loadtxt(prediction_file_path)
        assert len(predictions) == len(predictions[0])
        
        evals = {}
        evals['in_domain'] = in_domain(predictions)
        evals['next_domain'] = next_domain(predictions)
        evals['bwt'] = backward_transfer(predictions)
        evals['fwt'] = forward_transfer(predictions)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        score = np.array([evals[key] for key in evals.keys()]) @ weights

        # Plot accuracy matrix
        label_ticks = [str(i) for i in range(1, 11)]
        plot_2d_matrix(predictions, label_ticks, self.title, self.save_name, save_path=self.save_file_path)

        scores = {
            "score": score,
            "score_secondary": evals['next_domain'],
            "meta": {
                "in_domain_accuracy": evals['in_domain'],
                "backward_transfer": evals['bwt'],
                "forward_transfer": evals['fwt'],
            }
        }
        return scores
