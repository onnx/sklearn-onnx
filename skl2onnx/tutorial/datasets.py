"""
Some predefined datasets, predefined models.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from .. import to_onnx


def logreg_iris_onnx():
    data = load_iris()
    X, y = data.data, data.target
    logreg = LogisticRegression(solver="liblinear")
    logreg.fit(X, y)
    return to_onnx(logreg, X.astype(np.float32))
