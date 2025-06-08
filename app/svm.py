
import os
import pickle
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import image_dataset

class SvmEstimator(BaseEstimator): 

    def __init__(self):
        """
        Initialize an instance of our classifier
        """
        self.labels_ = None
        self.scaler = StandardScaler()
        self.model_ = SVC()
    
    def fit(self, X, y=None): 
        """
        Fit our SVM classifier to some data
        """
        self.labels_ = set(y)
        X_scaled = self.scaler_.fit_transform(X)
        self.model_.fit(X_scaled, y)

        return self

    def predict(self, X) -> np.ndarray: 
        """
        Predict classes for a set of inputs (images) based on a prior fit
        """
        X_scaled = self.scaler_.transform(X)        
        return self.model_.predict(X_scaled)
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """
        return np.mean(self.predict(X) == y)

def load_dataset(annotations, image_dir): 
    """
    Load and return a compatible dataset for the naive classifier
    """
    X = []
    y = list(annotations.label)

    for index, row in annotations.iterrows(): 
        file = image_dir + "/" + row.file
        X.append(image_dataset.load_image(file))

    return X, y

def eval(): 
    """
    Evaluate the model on a test set
    """
    pass 

def save_model(model:SvmEstimator, path):
    """
    Save the model to a file
    """    
    filename = os.path.join(path, 'svm.pkl')
    with open(filename, 'wb') as f: 
        pickle.dump(model, f)
    
    print(f"Model saved to {path}")
    return filename

def load_model(path) -> SvmEstimator: 
    """
    Load our classifier off disk
    """
    model = None

    filename = os.path.join(path, 'svm.pkl')
    with open(filename, 'r') as f: 
        model = pickle.load(f) 
    
    if type(model) != SVC: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

