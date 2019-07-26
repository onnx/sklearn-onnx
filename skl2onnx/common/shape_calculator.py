# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Functions to calculate output shapes of linear classifiers
and regressors.
"""
import numbers
import numpy as np
import six
from .data_types import (
    FloatTensorType, Int64TensorType, StringTensorType, DoubleTensorType
)
from .utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_linear_classifier_output_shapes(operator):
    """
    This operator maps an input feature vector into a scalar label if
    the number of outputs is one. If two outputs appear in this
    operator's output list, we should further generate a tensor storing
    all classes' probabilities.

    Allowed input/output patterns are
        1. [N, C] ---> [N, 1], A sequence of map

    """
    check_input_and_output_numbers(operator, input_count_range=1,
                                   output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[
        FloatTensorType, Int64TensorType, DoubleTensorType])

    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Inputs must be a [N, C]-tensor.')

    N = operator.inputs[0].type.shape[0]

    class_labels = operator.raw_operator.classes_
    number_of_classes = len(class_labels)
    if all(isinstance(i, np.ndarray) for i in class_labels):
        class_labels = np.concatenate(class_labels)
    if all(isinstance(i, (six.string_types, six.text_type))
           for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N])
        if number_of_classes > 2 or operator.type != 'SklearnLinearSVC':
            operator.outputs[1].type.shape = [N, number_of_classes]
        else:
            # For binary LinearSVC, we produce probability of
            # the positive class
            operator.outputs[1].type.shape = [N, 1]
    elif all(isinstance(i, (numbers.Real, bool, np.bool_))
             for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[N])
        if number_of_classes > 2 or operator.type != 'SklearnLinearSVC':
            operator.outputs[1].type.shape = [N, number_of_classes]
        else:
            # For binary LinearSVC, we produce probability of
            # the positive class
            operator.outputs[1].type.shape = [N, 1]
    else:
        raise ValueError('Label types must be all integers or all strings.')


def calculate_linear_regressor_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    This operator produces a scalar prediction for every example in a
    batch. If the input batch size is N, the output shape may be
    [N, 1].
    """
    check_input_and_output_numbers(operator, input_count_range=1,
                                   output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[
        FloatTensorType, Int64TensorType, DoubleTensorType])

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type.shape = [N, 1]
