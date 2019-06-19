# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy
from sklearn.base import ClusterMixin, BiclusterMixin, OutlierMixin
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC


def _problem_for_predictor_binary_classification():
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    binary classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    y[y == 2] = 1
    return (X, y, [('X', X[:1].astype(numpy.float32))],
            'predict_proba', 1, X.astype(numpy.float32))


def _problem_for_predictor_multi_classification():
    """
    Returns *X, y, intial_types, method, node name, X runtime* for a
    multi-class classification problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', X[:1].astype(numpy.float32))],
            'predict_proba', 1, X.astype(numpy.float32))


def _problem_for_predictor_regression():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    regression problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y.astype(float), [('X', X[:1].astype(numpy.float32))],
            'predict', 0, X.astype(numpy.float32))


def _problem_for_predictor_multi_regression():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    multi-regression problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target.astype(float)
    y2 = numpy.empty((y.shape[0], 2))
    y2[:, 0] = y
    y2[:, 1] = y + 0.5
    return (X, y2, [('X', X[:1].astype(numpy.float32))],
            'predict', 0, X.astype(numpy.float32))


def _problem_for_numerical_transform():
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(numpy.float32))],
            'transform', 0, X.astype(numpy.float32))


def _problem_for_numerical_trainable_transform():
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', X[:1].astype(numpy.float32))],
            'transform', 0, X.astype(numpy.float32))


def _problem_for_clustering():
    """
    Returns *X, intial_types, method, name, X runtime* for a
    clustering problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(numpy.float32))],
            'predict', 0, X.astype(numpy.float32))


def _problem_for_clustering_scores():
    """
    Returns *X, intial_types, method, name, X runtime* for a
    clustering problem, the score part, not the cluster.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(numpy.float32))],
            'transform', 1, X.astype(numpy.float32))


def _problem_for_outlier():
    """
    Returns *X, intial_types, method, name, X runtime* for a
    transformation problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    return (X, None, [('X', X[:1].astype(numpy.float32))],
            'predict', 0, X.astype(numpy.float32))


def _problem_for_numerical_scoring():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target.astype(float)
    y /= numpy.max(y)
    return (X, y, [('X', X[:1].astype(numpy.float32))],
            'score', 0, X.astype(numpy.float32))


def _problem_for_clnoproba():
    """
    Returns *X, y, intial_types, method, name, X runtime* for a
    scoring problem.
    It is based on Iris dataset.
    """
    data = load_iris()
    X = data.data
    y = data.target
    return (X, y, [('X', X[:1].astype(numpy.float32))],
            'predict', 0, X.astype(numpy.float32))


def find_suitable_problem(model):
    """
    Determines problems suitable for a given
    *scikit-learn* operator. It may be

    * `bin-class`: binary classification
    * `mutli-class`: multi-class classification
    * `regression`: regression
    * `multi-reg`: regression multi-output
    * `num-transform`: transform numerical features
    * `scoring`: transform numerical features, target is usually needed
    * `outlier`: outlier prediction
    * `linearsvc`: classifier without *predict_proba*
    * `cluster`: similar to transform
    * `num+y-trans`: similar to transform with targets
    * `num-trans-cluster`: similar to cluster, but returns
        scores or distances instead of cluster

    The following script gives the list of *scikit-learn*
    models and the problem they can be fitted on.
    """
    if model in {LinearSVC, NearestCentroid}:
        return ['clnoproba']
    if model in {RFE, RFECV, GridSearchCV}:
        return ['bin-class', 'multi-class',
                'regression', 'multi-reg',
                'cluster', 'outlier']
    if hasattr(model, 'predict_proba'):
        if model is OneVsRestClassifier:
            return ['multi-class']
        else:
            return ['bin-class', 'multi-class']

    if hasattr(model, 'predict'):
        if "Classifier" in str(model):
            return ['bin-class', 'multi-class']
        elif "Regressor" in str(model):
            return ['regression', 'multi-reg']

    res = []
    if hasattr(model, 'transform'):
        if issubclass(model, (RegressorMixin, ClassifierMixin)):
            res.extend(['num+y-trans'])
        elif issubclass(model, (ClusterMixin, BiclusterMixin)):
            res.extend(['num-trans-cluster'])
        else:
            res.extend(['num-transform'])

    if (hasattr(model, 'predict') and issubclass(model,
            (ClusterMixin, BiclusterMixin))):
        res.extend(['cluster'])

    if issubclass(model, (OutlierMixin)):
        res.extend(['outlier'])

    if issubclass(model, ClassifierMixin):
        res.extend(['bin-class', 'multi-class'])
    if issubclass(model, RegressorMixin):
        res.extend(['regression', 'multi-reg'])

    if len(res) == 0 and hasattr(model, 'fit') and hasattr(model, 'score'):
        return ['scoring']
    if len(res) > 0:
        return res

    raise RuntimeError("Unable to find problem for model '{}' - {}."
                       "".format(model.__name__, model.__bases__))


_problems = {
    "bin-class": _problem_for_predictor_binary_classification,
    "multi-class": _problem_for_predictor_multi_classification,
    "regression": _problem_for_predictor_regression,
    "multi-reg": _problem_for_predictor_multi_regression,
    "num-transform": _problem_for_numerical_transform,
    "scoring": _problem_for_numerical_scoring,
    'outlier': _problem_for_outlier,
    'clnoproba': _problem_for_clnoproba,
    'cluster': _problem_for_clustering,
    'num-trans-cluster': _problem_for_clustering_scores,
    'num+y-trans': _problem_for_numerical_trainable_transform,
}
