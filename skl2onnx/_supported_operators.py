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
try:
    from sklearn.ensemble import StackingClassifier, StackingRegressor
except ImportError:
    # New in 0.22
    StackingClassifier = None
    StackingRegressor = None
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
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
try:
    from sklearn.neighbors import (
        KNeighborsTransformer,
        NeighborhoodComponentsAnalysis,
    )
except ImportError:
    # New in 0.22
    KNeighborsTransformer = None
    NeighborhoodComponentsAnalysis = None

# Naive Bayes
from sklearn.naive_bayes import (
    BernoulliNB,
    GaussianNB,
    MultinomialNB,
)
try:
    from sklearn.naive_bayes import CategoricalNB
    from sklearn.naive_bayes import ComplementNB
except ImportError:
    # scikit-learn versions <= 0.21
    CategoricalNB = None
    # scikit-learn versions <= 0.19
    ComplementNB = None

# Neural Networks
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Clustering
from sklearn.cluster import KMeans, MiniBatchKMeans

# Operators for preprocessing and feature engineering
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import (
    PCA, IncrementalPCA, TruncatedSVD,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
from sklearn.feature_selection import (
    GenericUnivariateSelect, RFE, RFECV,
    SelectFdr, SelectFpr, SelectFromModel,
    SelectFwe, SelectKBest, SelectPercentile,
    VarianceThreshold
)
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
from sklearn.preprocessing import (
    LabelBinarizer, LabelEncoder,
    Normalizer, OneHotEncoder
)
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    # Not available in scikit-learn < 0.20.0
    OrdinalEncoder = None
from sklearn.preprocessing import (
    MinMaxScaler, MaxAbsScaler,
    FunctionTransformer,
    PolynomialFeatures, RobustScaler,
    StandardScaler, PowerTransformer,
)

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor
    )
except ImportError:
    # Second verification as these models still require
    # manual activation.
    try:
        from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (  # noqa
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor
        )
    except ImportError:
        HistGradientBoostingRegressor = None
        HistGradientBoostingClassifier = None

from .common._registration import register_converter, register_shape_calculator

# In most cases, scikit-learn operator produces only one output.
# However, each classifier has basically two outputs; one is the
# predicted label and the other one is the probabilities of all
# possible labels. Here is a list of supported scikit-learn
# classifiers. In the parsing stage, we produce two outputs for objects
# included in the following list and one output for everything not in
# the list.
sklearn_classifier_list = list(filter(lambda m: m is not None, [
    LogisticRegression, LogisticRegressionCV, Perceptron, SGDClassifier,
    PassiveAggressiveClassifier,
    LinearSVC, SVC, NuSVC,
    GradientBoostingClassifier, RandomForestClassifier,
    DecisionTreeClassifier, ExtraTreeClassifier, ExtraTreesClassifier,
    BaggingClassifier, StackingClassifier,
    BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB,
    KNeighborsClassifier,
    CalibratedClassifierCV, OneVsRestClassifier, VotingClassifier,
    AdaBoostClassifier, MLPClassifier, LinearDiscriminantAnalysis,
    HistGradientBoostingClassifier
]))

# Clustering algorithms: produces two outputs, label and score for
# each cluster in most cases.
cluster_list = [KMeans, MiniBatchKMeans]

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
                CategoricalNB,
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
                StackingClassifier, StackingRegressor,
                VotingClassifier, VotingRegressor,
                KMeans, MiniBatchKMeans,
                NeighborhoodComponentsAnalysis,
                PCA, TruncatedSVD, IncrementalPCA,
                Binarizer, MinMaxScaler, MaxAbsScaler, Normalizer,
                CountVectorizer, TfidfVectorizer, TfidfTransformer,
                FunctionTransformer, KBinsDiscretizer, PolynomialFeatures,
                Imputer, SimpleImputer, LabelBinarizer, LabelEncoder,
                KNeighborsTransformer,
                RobustScaler, OneHotEncoder, DictVectorizer, OrdinalEncoder,
                GenericUnivariateSelect, RFE, RFECV, SelectFdr, SelectFpr,
                SelectFromModel, SelectFwe, SelectKBest, SelectPercentile,
                VarianceThreshold, GaussianMixture, GaussianProcessRegressor,
                BayesianGaussianMixture, OneClassSVM, PLSRegression,
                HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                PowerTransformer,
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
                                overwrite=True, parser=None, options=None):
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
    :param options: registered options for this converter

    The alias is usually the library name followed by the model name.
    Example:

    ::

        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from skl2onnx.operator_converters.RandomForest import convert_sklearn_random_forest_classifier
        from skl2onnx import update_registered_converter
        update_registered_converter(SGDClassifier, 'SklearnLinearClassifier',
                                    calculate_linear_classifier_output_shapes,
                                    convert_sklearn_random_forest_classifier,
                                    options={'zipmap': [True, False],
                                             'raw_scores': [True, False]})
    """ # noqa
    if (not overwrite and model in sklearn_operator_name_map
            and alias != sklearn_operator_name_map[model]):
        warnings.warn("Model '{0}' was already registered under alias "
                      "'{1}'.".format(model, sklearn_operator_name_map[model]))
    sklearn_operator_name_map[model] = alias
    register_converter(alias, convert_fct, overwrite=overwrite,
                       options=options)
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
