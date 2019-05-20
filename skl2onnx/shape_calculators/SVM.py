# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from sklearn.svm import SVC, NuSVC
from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, Int64TensorType
from ..common.data_types import StringTensorType
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_svm_output_shapes(operator):
    """
    For SVM classifiers, allowed input/output patterns are
        1. [N, C] ---> [N], A sequence of map
    Note that the second case is not allowed as long as ZipMap only
    produces dictionary.

    For SVM regressors, allowed input/output patterns are
        1. [N, C] ---> [N]

    For both of SVC and SVR, the inputs should numerical tensor(s).
    For SVC with batch size 1, the first output is the label and the
    second output is a map used to store all class probabilities (For
    a key-value pair, the value is assigned to the class specified by
    the key). If batch size is larger than 1, we need to use a sequence
    of maps to denote class probabilities. Regarding SVR, we just
    produce a scalar for each example. If there are N examples, the
    output shape would be [N, 1].
    """
    op = operator.raw_operator

    N = operator.inputs[0].type.shape[0]
    if operator.type in ['SklearnSVC'] or isinstance(op, (SVC, NuSVC)):
        number_of_classes = len(op.classes_)
        check_input_and_output_numbers(operator, input_count_range=[1, None],
                                       output_count_range=[1, 2])

        if all(isinstance(i, (six.string_types, six.text_type))
               for i in op.classes_):
            operator.outputs[0].type = StringTensorType([N])
            operator.outputs[1].type = FloatTensorType([N, number_of_classes])
        elif all(isinstance(i, (numbers.Real, bool, np.bool_))
                 for i in op.classes_):
            operator.outputs[0].type = Int64TensorType([N])
            operator.outputs[1].type = FloatTensorType([N, number_of_classes])
        else:
            raise RuntimeError('Class labels should be either all strings or '
                               'all integers')

    if operator.type in ['SklearnSVR']:
        check_input_and_output_numbers(operator, input_count_range=[1, None],
                                       output_count_range=1)

        operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('SklearnSVC', calculate_sklearn_svm_output_shapes)
register_shape_calculator('SklearnSVR', calculate_sklearn_svm_output_shapes)
