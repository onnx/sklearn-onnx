# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register a converter for scikit-learn operators,
# import associated modules here.
from . import ada_boost
from . import array_feature_extractor
from . import binariser
from . import calibrated_classifier_cv
from . import concat_op
from . import decision_tree
from . import decomposition
from . import dict_vectoriser
from . import feature_selection
from . import flatten_op
from . import function_transformer
from . import gradient_boosting
from . import imputer_op
from . import k_bins_discretiser
from . import k_means
from . import label_binariser
from . import label_encoder
from . import linear_classifier
from . import linear_regressor
from . import multilayer_perceptron
from . import naive_bayes
from . import nearest_neighbours
from . import normaliser
from . import one_hot_encoder
from . import one_vs_rest_classifier
from . import polynomial_features
from . import random_forest
from . import scaler_op
from . import sgd_classifier
from . import support_vector_machines
from . import text_vectoriser
from . import tfidf_transformer
from . import voting_classifier
from . import zip_map

__all__ = [
    ada_boost,
    array_feature_extractor,
    binariser,
    calibrated_classifier_cv,
    concat_op,
    decision_tree,
    decomposition,
    dict_vectoriser,
    feature_selection,
    flatten_op,
    function_transformer,
    gradient_boosting,
    imputer_op,
    k_bins_discretiser,
    k_means,
    label_binariser,
    label_encoder,
    linear_classifier,
    linear_regressor,
    multilayer_perceptron,
    naive_bayes,
    nearest_neighbours,
    normaliser,
    one_hot_encoder,
    one_vs_rest_classifier,
    polynomial_features,
    random_forest,
    scaler_op,
    sgd_classifier,
    support_vector_machines,
    text_vectoriser,
    tfidf_transformer,
    voting_classifier,
    zip_map,
]
