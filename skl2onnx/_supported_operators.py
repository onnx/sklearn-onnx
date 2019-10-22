# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import warnings

# Calibrated classifier CV
from sklearn.calibration import CalibratedClassifierCV

# Linear classifiers
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron, SGDClassifier,
    RidgeClassifier, RidgeClassifierCV,
)
from sklearn.svm import LinearSVC, OneClassSVM

# Linear regressors
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet, ElasticNetCV,
    HuberRegressor,
    Lars, LarsCV,
    Lasso, LassoCV,
    LassoLars, LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    MultiTaskElasticNet, MultiTaskElasticNetCV,
    MultiTaskLasso, MultiTaskLassoCV,
    OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge, RidgeCV,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.svm import LinearSVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Mixture
from sklearn.mixture import (
    GaussianMixture, BayesianGaussianMixture
)

# Multi-class
from sklearn.multiclass import OneVsRestClassifier

# Tree-based models
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier
)
try:
    from sklearn.ensemble import VotingRegressor
except ImportError:
    # New in 0.21
    VotingRegressor = None
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    ExtraTreeClassifier, ExtraTreeRegressor
)

# Gaussian processes
from sklearn.gaussian_process import GaussianProcessRegressor

# GridSearchCV
from sklearn.model_selection import GridSearchCV

# Support vector machines
from sklearn.svm import NuSVC, NuSVR, SVC, SVR

# K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors

# Naive Bayes
from sklearn.naive_bayes import (
    BernoulliNB,
    GaussianNB,
    MultinomialNB,
)
try:
    from sklearn.naive_bayes import ComplementNB
except ImportError:
    # scikit-learn versions <= 0.19
    ComplementNB = None

# Neural Networks
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Clustering
from sklearn.cluster import KMeans, MiniBatchKMeans

# Operators for preprocessing and feature engineering
from sklearn.decomposition import (
    PCA, IncrementalPCA, TruncatedSVD
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import GenericUnivariateSelect, RFE, RFECV
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFromModel
from sklearn.feature_selection import SelectFwe, SelectKBest, SelectPercentile
from sklearn.feature_selection import VarianceThreshold
try:
    # 0.20
    from sklearn.impute import SimpleImputer
except ImportError:
    # 0.19
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.preprocessing import Binarizer
try:
    from sklearn.preprocessing import Imputer
except ImportError:
    # removed in 0.21
    Imputer = None
try:
    from sklearn.preprocessing import KBinsDiscretizer
except ImportError:
    # not available in 0.19
    KBinsDiscretizer = None
    pass
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    # Not available in scikit-learn < 0.20.0
    OrdinalEncoder = None
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import FunctionTransformer

from .common._registration import register_converter, register_shape_calculator

# In most cases, scikit-learn operator produces only one output.
# However, each classifier has basically two outputs; one is the
# predicted label and the other one is the probabilities of all
# possible labels. Here is a list of supported scikit-learn
# classifiers. In the parsing stage, we produce two outputs for objects
# included in the following list and one output for everything not in
# the list.
sklearn_classifier_list = [
    LogisticRegression, LogisticRegressionCV, Perceptron, SGDClassifier,
    PassiveAggressiveClassifier,
    LinearSVC, SVC, NuSVC,
    GradientBoostingClassifier, RandomForestClassifier,
    DecisionTreeClassifier, ExtraTreeClassifier, ExtraTreesClassifier,
    BaggingClassifier,
    BernoulliNB, ComplementNB, GaussianNB, MultinomialNB,
    KNeighborsClassifier,
    CalibratedClassifierCV, OneVsRestClassifier, VotingClassifier,
    AdaBoostClassifier, MLPClassifier, LinearDiscriminantAnalysis
]

# Clustering algorithms: produces two outputs, label and score for
# each cluster in most cases.
cluster_list = [KMeans, MiniBatchKMeans]

# Classifiers with converters supporting decision_function().
decision_function_classifiers = (
    SGDClassifier,
)

# Outlier detection algorithms:
# produces two outputs, label and scores
outlier_list = [OneClassSVM]


# Associate scikit-learn types with our operator names. If two
# scikit-learn models share a single name, it means their are
# equivalent in terms of conversion.
def build_sklearn_operator_name_map():
    res = {k: "Sklearn" + k.__name__ for k in [
                AdaBoostClassifier, AdaBoostRegressor,
                BaggingClassifier, BaggingRegressor,
                BernoulliNB, ComplementNB, GaussianNB, MultinomialNB,
                CalibratedClassifierCV,
                DecisionTreeClassifier, DecisionTreeRegressor,
                ExtraTreeClassifier, ExtraTreeRegressor,
                ExtraTreesClassifier, ExtraTreesRegressor,
                GradientBoostingClassifier, GradientBoostingRegressor,
                KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors,
                LinearSVC, LinearSVR, SVC, SVR,
                LinearRegression, RANSACRegressor,
                MLPClassifier, MLPRegressor,
                OneVsRestClassifier,
                RandomForestClassifier, RandomForestRegressor,
                SGDClassifier,
                VotingClassifier, VotingRegressor,
                KMeans, MiniBatchKMeans,
                PCA, TruncatedSVD, IncrementalPCA,
                Binarizer, MinMaxScaler, MaxAbsScaler, Normalizer,
                CountVectorizer, TfidfVectorizer, TfidfTransformer,
                FunctionTransformer, KBinsDiscretizer, PolynomialFeatures,
                Imputer, SimpleImputer, LabelBinarizer, LabelEncoder,
                RobustScaler, OneHotEncoder, DictVectorizer, OrdinalEncoder,
                GenericUnivariateSelect, RFE, RFECV, SelectFdr, SelectFpr,
                SelectFromModel, SelectFwe, SelectKBest, SelectPercentile,
                VarianceThreshold, GaussianMixture, GaussianProcessRegressor,
                BayesianGaussianMixture, OneClassSVM
    ] if k is not None}
    res.update({
        ARDRegression: 'SklearnLinearRegressor',
        BayesianRidge: 'SklearnLinearRegressor',
        ElasticNet: 'SklearnLinearRegressor',
        ElasticNetCV: 'SklearnLinearRegressor',
        GridSearchCV: 'SklearnGridSearchCV',
        HuberRegressor: 'SklearnLinearRegressor',
        LinearRegression: 'SklearnLinearRegressor',
        Lars: 'SklearnLinearRegressor',
        LarsCV: 'SklearnLinearRegressor',
        Lasso: 'SklearnLinearRegressor',
        LassoCV: 'SklearnLinearRegressor',
        LassoLars: 'SklearnLinearRegressor',
        LassoLarsCV: 'SklearnLinearRegressor',
        LassoLarsIC: 'SklearnLinearRegressor',
        LinearDiscriminantAnalysis: 'SklearnLinearClassifier',
        LogisticRegression: 'SklearnLinearClassifier',
        LogisticRegressionCV: 'SklearnLinearClassifier',
        MultiTaskElasticNet: 'SklearnLinearRegressor',
        MultiTaskElasticNetCV: 'SklearnLinearRegressor',
        MultiTaskLasso: 'SklearnLinearRegressor',
        MultiTaskLassoCV: 'SklearnLinearRegressor',
        NuSVC: 'SklearnSVC',
        NuSVR: 'SklearnSVR',
        OrthogonalMatchingPursuit: 'SklearnLinearRegressor',
        OrthogonalMatchingPursuitCV: 'SklearnLinearRegressor',
        PassiveAggressiveClassifier: 'SklearnSGDClassifier',
        PassiveAggressiveRegressor: 'SklearnLinearRegressor',
        Perceptron: 'SklearnSGDClassifier',
        Ridge: 'SklearnLinearRegressor',
        RidgeCV: 'SklearnLinearRegressor',
        RidgeClassifier: 'SklearnLinearClassifier',
        RidgeClassifierCV: 'SklearnLinearClassifier',
        SGDRegressor: 'SklearnLinearRegressor',
        StandardScaler: 'SklearnScaler',
        TheilSenRegressor: 'SklearnLinearRegressor',
    })
    return res


def update_registered_converter(model, alias, shape_fct, convert_fct,
                                overwrite=True, parser=None):
    """
    Registers or updates a converter for a new model so that
    it can be converted when inserted in a *scikit-learn* pipeline.

    :param model: model class
    :param alias: alias used to register the model
    :param shape_fct: function which checks or modifies the expected
        outputs, this function should be fast so that the whole graph
        can be computed followed by the conversion of each model,
        parallelized or not
    :param convert_fct: function which converts a model
    :param overwrite: False to raise exception if a converter
        already exists
    :param parser: overwrites the parser as well if not empty

    The alias is usually the library name followed by the model name.
    Example:

    ::

        from onnxmltools.convert.common.shape_calculator import calculate_linear_classifier_output_shapes
        from skl2onnx.operator_converters.RandomForest import convert_sklearn_random_forest_classifier
        from skl2onnx import update_registered_converter
        update_registered_converter(SGDClassifier, 'SklearnLinearClassifier',
                                    calculate_linear_classifier_output_shapes,
                                    convert_sklearn_random_forest_classifier)
    """ # noqa
    if (not overwrite and model in sklearn_operator_name_map
            and alias != sklearn_operator_name_map[model]):
        warnings.warn("Model '{0}' was already registered under alias "
                      "'{1}'.".format(model, sklearn_operator_name_map[model]))
    sklearn_operator_name_map[model] = alias
    register_converter(alias, convert_fct, overwrite=overwrite)
    register_shape_calculator(alias, shape_fct, overwrite=overwrite)
    if parser is not None:
        from ._parse import update_registered_parser
        update_registered_parser(model, parser)


def _get_sklearn_operator_name(model_type):
    """
    Get operator name of the input argument

    :param model_type:  A scikit-learn object (e.g., SGDClassifier
                        and Binarizer)
    :return: A string which stands for the type of the input model in
             our conversion framework
    """
    if model_type not in sklearn_operator_name_map:
        # "No proper operator name found, it means a local operator.
        return None
    return sklearn_operator_name_map[model_type]


# registered converters
sklearn_operator_name_map = build_sklearn_operator_name_map()
