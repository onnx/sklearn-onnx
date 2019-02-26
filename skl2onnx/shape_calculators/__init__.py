# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register shape calculators for scikit-learn operators,
# import associated modules here.
from . import ArrayFeatureExtractor
from . import Concat
from . import DictVectorizer
from . import Flatten
from . import FunctionTransformer
from . import Imputer
from . import KBinsDiscretiser
from . import KMeans
from . import LabelBinariser
from . import LabelEncoder
from . import LinearClassifier
from . import LinearRegressor
from . import NearestNeighbours
from . import OneHotEncoder
from . import OneVsRestClassifier
from . import PolynomialFeatures
from . import Scaler
from . import SVD
from . import SVM
from . import TextVectorizer
from . import TfidfTransformer
from . import VotingClassifier
from . import ZipMap

__all__ = [
    ArrayFeatureExtractor,
    Concat,
    DictVectorizer,
    FunctionTransformer,
    Imputer,
    KBinsDiscretiser,
    KMeans,
    LabelBinariser,
    LabelEncoder,
    LinearClassifier,
    LinearRegressor,
    NearestNeighbours,
    OneHotEncoder,
    OneVsRestClassifier,
    PolynomialFeatures,
    Scaler,
    SVD,
    SVM,
    TextVectorizer,
    TfidfTransformer,
    VotingClassifier,
    ZipMap,
]
