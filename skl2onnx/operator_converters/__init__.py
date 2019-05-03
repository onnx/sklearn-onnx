# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register a converter for scikit-learn operators,
# import associated modules here.
from . import AdaBoost
from . import ArrayFeatureExtractor
from . import Binarizer
from . import CalibratedClassifierCV
from . import Concat
from . import DecisionTree
from . import DictVectorizer
from . import FeatureSelection
from . import Flatten
from . import FunctionTransformer
from . import GradientBoosting
from . import Imputer
from . import KBinsDiscretiser
from . import KMeans
from . import KNN
from . import LabelBinariser
from . import LabelEncoder
from . import LinearClassifier
from . import LinearRegressor
from . import multilayer_perceptron
from . import NaiveBayes
from . import Normalizer
from . import OneHotEncoder
from . import OneVsRestClassifier
from . import PolynomialFeatures
from . import RandomForest
from . import Scaler
from . import sgd_classifier
from . import SVD
from . import SVM
from . import TextVectorizer
from . import TfIdfTransformer
from . import VotingClassifier
from . import ZipMap

__all__ = [
    AdaBoost,
    ArrayFeatureExtractor,
    Binarizer,
    CalibratedClassifierCV,
    Concat,
    DecisionTree,
    DictVectorizer,
    FeatureSelection,
    Flatten,
    FunctionTransformer,
    GradientBoosting,
    Imputer,
    KBinsDiscretiser,
    KMeans,
    KNN,
    LabelBinariser,
    LabelEncoder,
    LinearClassifier,
    LinearRegressor,
    multilayer_perceptron,
    NaiveBayes,
    Normalizer,
    OneHotEncoder,
    OneVsRestClassifier,
    PolynomialFeatures,
    RandomForest,
    Scaler,
    sgd_classifier,
    SVD,
    SVM,
    TextVectorizer,
    TfIdfTransformer,
    VotingClassifier,
    ZipMap,
]
