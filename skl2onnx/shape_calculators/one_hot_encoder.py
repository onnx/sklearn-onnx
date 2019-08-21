# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._registration import register_shape_calculator


def calculate_sklearn_one_hot_encoder_output_shapes(operator):
    op = operator.raw_operator
    categories_len = 0
    for index, categories in enumerate(op.categories_):
        if hasattr(op, 'drop_idx_') and op.drop_idx_ is not None:
            categories = (categories[np.arange(len(categories)) !=
                          op.drop_idx_[index]])
        categories_len += len(categories)
    instances = operator.inputs[0].type.shape[0]
    operator.outputs[0].type.shape = [instances, categories_len]


register_shape_calculator('SklearnOneHotEncoder',
                          calculate_sklearn_one_hot_encoder_output_shapes)
