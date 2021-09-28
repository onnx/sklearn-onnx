# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import numpy
import onnx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.proto import onnx_proto
from skl2onnx.common._apply_operation import apply_mul
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_multiple_classification,
    dump_binary_classification,
    dump_data_and_model,
    TARGET_OPSET
)


class CustomTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X, y, sample_weight=None):
        pass

    def transform(self, X):
        return X * numpy.array([[0.5, 0.1, 10], [0.5, 0.1, 10]]).T


def custom_transform_shape_calculator(operator):
    operator.outputs[0].type = FloatTensorType([3, 2])


def custom_tranform_converter(scope, operator, container):
    input = operator.inputs[0]
    output = operator.outputs[0]

    weights_name = scope.get_unique_variable_name("weights")
    atype = onnx_proto.TensorProto.FLOAT
    weights = [0.5, 0.1, 10]
    shape = [len(weights), 1]
    container.add_initializer(weights_name, atype, shape, weights)
    apply_mul(scope, [input.full_name, weights_name], output.full_name,
              container)


class TestVotingClassifierConverter(unittest.TestCase):
    def test_operator_mul(self):

        model = CustomTransform()
        Xd = numpy.array([[1, 2], [3, 4], [4, 5]])

        model_onnx = convert_sklearn(
            model, "CustomTransform",
            [("input", FloatTensorType([None, Xd.shape[1]]))],
            custom_shape_calculators={
                CustomTransform: custom_transform_shape_calculator
            },
            custom_conversion_functions={
                CustomTransform: custom_tranform_converter
            }, target_opset=TARGET_OPSET)
        dump_data_and_model(
            Xd.astype(numpy.float32), model, model_onnx,
            basename="CustomTransformerMul")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
                     reason="not available")
    def test_voting_hard_binary(self):
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        # predict_proba is not defined when voting is hard.
        dump_binary_classification(
            model, suffix="Hard", comparable_outputs=[0],
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
                     reason="not available")
    def test_voting_hard_binary_weights(self):
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            weights=numpy.array([1000, 1]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        # predict_proba is not defined when voting is hard.
        dump_binary_classification(
            model, suffix="WeightsHard", comparable_outputs=[0],
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_voting_soft_binary(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        dump_binary_classification(
            model, suffix="Soft", comparable_outputs=[0, 1],
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_voting_soft_binary_weighted(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([1.8, 0.2]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        dump_binary_classification(
            model, suffix="WeightedSoft",
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
                     reason="not available")
    def test_voting_hard_multi(self):
        # predict_proba is not defined when voting is hard.
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", DecisionTreeClassifier()),
            ],
        )
        dump_multiple_classification(
            model, suffix="Hard", comparable_outputs=[0],
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
                     reason="not available")
    def test_voting_hard_multi_weighted(self):
        # predict_proba is not defined when voting is hard.
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            weights=numpy.array([1000, 1]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", DecisionTreeClassifier()),
            ],
        )
        dump_multiple_classification(
            model, suffix="WeightedHard", comparable_outputs=[0],
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_voting_soft_multi(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, suffix="Soft", target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_voting_soft_multi_string(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, label_string=True, suffix="Soft",
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_voting_soft_multi_weighted(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([1.8, 0.2]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, suffix="WeightedSoft",
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_voting_soft_multi_weighted4(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([2.7, 0.3, 0.5, 0.5]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lra", LogisticRegression()),
                ("lrb", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, suffix="Weighted4Soft",
            target_opset=TARGET_OPSET)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_voting_soft_multi_weighted42(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([27, 0.3, 0.5, 0.5]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lra", LogisticRegression()),
                ("lrb", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, suffix="Weighted42Soft",
            target_opset=TARGET_OPSET)


if __name__ == "__main__":
    unittest.main()
