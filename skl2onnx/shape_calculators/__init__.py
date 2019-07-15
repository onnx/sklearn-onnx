# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register shape calculators for scikit-learn operators,
# import associated modules here.
from . import array_feature_extractor
from . import concat
from . import dict_vectorizer
from . import flatten
from . import function_transformer
from . import imputer
from . import k_bins_discretiser
from . import k_means
from . import label_binariser
from . import label_encoder
from . import linear_classifier
from . import linear_regressor
from . import mixture
from . import nearest_neighbours
from . import one_hot_encoder
from . import one_vs_rest_classifier
from . import polynomial_features
from . import scaler
from . import svd
from . import svm
from . import text_vectorizer
from . import tfidf_transformer
from . import voting_classifier
from . import zip_map

__all__ = [
    array_feature_extractor,
    concat,
    dict_vectorizer,
    flatten,
    function_transformer,
    imputer,
    k_bins_discretiser,
    k_means,
    label_binariser,
    label_encoder,
    linear_classifier,
    linear_regressor,
    mixture,
    nearest_neighbours,
    one_hot_encoder,
    one_vs_rest_classifier,
    polynomial_features,
    scaler,
    svd,
    svm,
    text_vectorizer,
    tfidf_transformer,
    voting_classifier,
    zip_map,
]
