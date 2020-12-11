# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register a converter for scikit-learn operators,
# import associated modules here.
from . import ada_boost
from . import array_feature_extractor
from . import bagging
from . import binariser
from . import calibrated_classifier_cv
from . import cast_op
from . import concat_op
from . import cross_decomposition
from . import decision_tree
from . import decomposition
from . import dict_vectoriser
from . import feature_selection
from . import flatten_op
from . import function_transformer
from . import gaussian_process
from . import gaussian_mixture
from . import gradient_boosting
from . import grid_search_cv
from . import id_op
from . import imputer_op
from . import isolation_forest
from . import k_bins_discretiser
from . import k_means
from . import label_binariser
from . import label_encoder
from . import linear_classifier
from . import linear_regressor
from . import multilayer_perceptron
from . import multiply_op
from . import naive_bayes
from . import nearest_neighbours
from . import normaliser
from . import one_hot_encoder
from . import one_vs_rest_classifier
from . import ordinal_encoder
from . import polynomial_features
from . import power_transformer
from . import random_forest
from . import random_projection
from . import ransac_regressor
from . import scaler_op
from . import sgd_classifier
from . import stacking
from . import support_vector_machines
from . import text_vectoriser
from . import tfidf_transformer
from . import tfidf_vectoriser
from . import voting_classifier
from . import voting_regressor
from . import zip_map

__all__ = [
    ada_boost,
    array_feature_extractor,
    bagging,
    binariser,
    calibrated_classifier_cv,
    cast_op,
    concat_op,
    cross_decomposition,
    decision_tree,
    decomposition,
    dict_vectoriser,
    feature_selection,
    flatten_op,
    function_transformer,
    gaussian_process,
    gaussian_mixture,
    gradient_boosting,
    grid_search_cv,
    id_op,
    imputer_op,
    isolation_forest,
    k_bins_discretiser,
    k_means,
    label_binariser,
    label_encoder,
    linear_classifier,
    linear_regressor,
    multilayer_perceptron,
    multiply_op,
    naive_bayes,
    nearest_neighbours,
    normaliser,
    one_hot_encoder,
    one_vs_rest_classifier,
    ordinal_encoder,
    polynomial_features,
    power_transformer,
    random_forest,
    random_projection,
    ransac_regressor,
    scaler_op,
    sgd_classifier,
    stacking,
    support_vector_machines,
    text_vectoriser,
    tfidf_transformer,
    tfidf_vectoriser,
    voting_classifier,
    voting_regressor,
    zip_map,
]
