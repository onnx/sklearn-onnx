# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType, StringTensorType


def calculate_quadratic_discriminant_analysis_shapes(operator):
    classes = operator.raw_operator.classes_
    if all((isinstance(s, str)) for s in classes):
        label_tensor_type = StringTensorType
    else:
        label_tensor_type = Int64TensorType

    n_clasess = len(classes)
    operator.outputs[0].type = label_tensor_type([1, None])
    operator.outputs[1].type.shape = [None, n_clasess]


register_shape_calculator(
    "SklearnQuadraticDiscriminantAnalysis",
    calculate_quadratic_discriminant_analysis_shapes,
)
