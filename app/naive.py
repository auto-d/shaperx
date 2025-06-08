import cv2
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
import matplotlib.pyplot as plt
import pickle

class ShapeRxEstimator(BaseEstimator): 
    """
    Interface for loading, fitting and predicting on images in our dataset. 
    This is really about enforcing semantics to ensure we can trivially 
    run cross-validation and hyperparameter search over the implementing 
    models with e.g. sklearn. 
    """
    columns = ['source', 'file', 'label']

    def fit(self, X, y=None):
        """
        Fit our estimator to some images. Note we may want to 
        pass this model into a grid search and repeatedly fit on subsets
        to identify optimal hyperparameters, so we need to handle
        reloading the training data here every time the method is called
        """
        raise NotImplementedError("Missing implementation for fit()!")

    def predict(self, X) -> np.ndarray: 
        """
        Make a prediction on a single image
        """
        raise NotImplementedError("Missing implementation for predict()!")

class NaiveEstimator(ShapeRxEstimator): 

    def __init__(self, bins=256):
        """
        Set up an instance of our naive histogram-based estimator. Sklearn requires
        us to do no magic here. Just store passed params or we break cloning and 
        reproducibility. 
        """
        self.bins = bins
        self.histograms_ = None
        self.labels_ = None
    
    def fit(self, X, y=None): 
        """
        Fit our naive estimator to a training set (of images) ... SKLearn doesn't 
        support batching so we don't either. Our dataset will fit in memory and we'll 
        stick to that convention for these projects to allow for joint search of 
        classic methods and neural methods in the same SKL pipelines. 
        """
        labels = set(y)

        #TODO: this will raise if y is not contiguous -- I don't think we need to 
        # add more abstraction here given we control the whole operation and expect to 
        # provide examples of every class, coded as 0 - 24. 
        pixel_counters = np.zeros((len(labels), self.bins))
        class_counters = np.zeros((len(labels)))
        
        # Tally pixels in each bin for every class, keeping track of the 
        # number of class instances we see
        for image, label in zip(X,y):  
            hist = self.histogram(image) 

            pixel_counters[label] = pixel_counters[label] + hist
            class_counters[label] += 1

        # Compute mean 
        for label in labels: 
            if class_counters[label] > 0: 
                pixel_counters[label] = pixel_counters[label] / class_counters[label]

        # Sklearn expects us to store learned params with a trailing underscore
        self.labels_ = labels
        self.histograms_ = pixel_counters.astype(np.float32)

        return self

    def predict(self, X) -> np.ndarray: 
        """
        Predict classes for a set of inputs (images) based on a prior fit
        """
        preds = np.zeros(len(X))
        
        for i, x in enumerate(X): 
            shortest_distance = None
            hist = self.histogram(x)

            # Apply a chi-squared distance measure to identify the closest match to our input
            for label in self.labels_: 
                distance = cv2.compareHist(hist, self.histograms_[label], method=cv2.HISTCMP_CHISQR)
                if shortest_distance is None or distance < shortest_distance: 
                    shortest_distance = distance
                    preds[i] = label
        
        return preds 

    def histogram(self, image:np.ndarray=None): 
        """
        Generate a histogram of the provided image

        With help from openCV docs: https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
        """
        hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[self.bins], ranges=[0,256])
        return hist.flatten()
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """
        return np.mean(self.predict(X) == y)


def eval(X): 
    """
    Evaluate the model on a test set
    """


    pass 

def save_model(model:NaiveEstimator, path):
    """
    Save the model to a file
    """    
    filename = os.path.join(path, 'naive.pkl')
    with open(filename, 'wb') as f: 
        pickle.dump(model, f)
    
    print(f"Model saved to {path}")

def load_model(path): 
    """
    Load our naive model off disk
    """
    model = None

    filename = os.path.join(path, 'naive.pkl')
    with open(filename, 'rb') as f: 
        model = pickle.load(f) 
    
    if type(model) != NaiveEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model