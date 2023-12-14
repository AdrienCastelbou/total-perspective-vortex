import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin

class SingleClassTransf(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        return X.reshape(X.shape[0], -1)
    
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)