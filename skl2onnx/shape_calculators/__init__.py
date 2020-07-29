# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register shape calculators for scikit-learn operators,
# import associated modules here.
from . import array_feature_extractor
from . import cast_op
from . import concat
from . import cross_decomposition
from . import dict_vectorizer
from . import ensemble_shapes
from . import flatten
from . import function_transformer
from . import gaussian_process
from . import grid_search_cv
from . import identity
from . import imputer
from . import isolation_forest
from . import k_bins_discretiser
from . import k_means
from . import label_binariser
from . import label_encoder
from . import linear_classifier
from . import linear_regressor
from . import mixture
from . import nearest_neighbours
from . import one_hot_encoder
from . import ordinal_encoder
from . import one_vs_rest_classifier
from . import polynomial_features
from . import power_transformer
from . import random_projection
from . import scaler
from . import svd
from . import support_vector_machines
from . import text_vectorizer
from . import tfidf_transformer
from . import voting_classifier
from . import voting_regressor
from . import zip_map

__all__ = [
    array_feature_extractor,
    cast_op,
    concat,
    cross_decomposition,
    dict_vectorizer,
    ensemble_shapes,
    flatten,
    function_transformer,
    gaussian_process,
    grid_search_cv,
    identity,
    imputer,
    isolation_forest,
    k_bins_discretiser,
    k_means,
    label_binariser,
    label_encoder,
    linear_classifier,
    linear_regressor,
    mixture,
    nearest_neighbours,
    one_hot_encoder,
    ordinal_encoder,
    one_vs_rest_classifier,
    polynomial_features,
    power_transformer,
    random_projection,
    scaler,
    svd,
    support_vector_machines,
    text_vectorizer,
    tfidf_transformer,
    voting_classifier,
    voting_regressor,
    zip_map,
]
