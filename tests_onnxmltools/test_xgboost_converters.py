# SPDX-License-Identifier: Apache-2.0


import unittest
import numbers
import packaging.version as pv
import numpy as np
from numpy.testing import assert_almost_equal
import pandas
from onnxruntime import InferenceSession
from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

try:
    from sklearn.ensemble import StackingClassifier
except ImportError:
    # New in 0.22
    StackingClassifier = None
from skl2onnx import update_registered_converter, convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,  # noqa
    calculate_linear_regressor_output_shapes,
)
from skl2onnx._parse import _parse_sklearn_classifier
from xgboost import XGBRegressor, XGBClassifier
import onnxmltools
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost,  # noqa
)

try:
    from test_utils import dump_single_regression
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
    from test_utils import dump_single_regression
from test_utils import dump_multiple_classification, TARGET_OPSET, TARGET_OPSET_ML


class TestXGBoostModels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        def custom_parser(scope, model, inputs, custom_parsers=None):
            if custom_parsers is not None and model in custom_parsers:
                return custom_parsers[model](
                    scope, model, inputs, custom_parsers=custom_parsers
                )
            if not all(
                isinstance(i, (numbers.Real, bool, np.bool_)) for i in model.classes_
            ):
                raise NotImplementedError(
                    "Current converter does not support string labels."
                )
            return _parse_sklearn_classifier(scope, model, inputs)

        update_registered_converter(
            XGBClassifier,
            "XGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            parser=custom_parser,
            options={"zipmap": [True, False, "columns"], "nocl": [True, False]},
        )
        update_registered_converter(
            XGBRegressor,
            "XGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost,
        )

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version("1.12"),
        reason="converter for xgboost is too old",
    )
    def test_xgb_regressor(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBRegressor()
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, X.shape[1]]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        self.assertTrue(conv_model is not None)
        dump_single_regression(
            xgb,
            suffix="-Dec4",
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )

    def test_xgb_classifier(self):
        xgb = XGBClassifier(n_estimators=2, max_depth=2)
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, X.shape[1]]))],
            options={id(xgb): {"zipmap": False}},
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        sess = InferenceSession(
            conv_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": X.astype(np.float32)})
        assert_almost_equal(xgb.predict_proba(X), res[1])
        assert_almost_equal(xgb.predict(X), res[0])

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version("1.11"),
        reason="converter for xgboost is too old",
    )
    def test_xgb_classifier_multi(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier()
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, X.shape[1]]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(
            xgb, target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML}
        )

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version("1.11"),
        reason="converter for xgboost is too old",
    )
    def test_xgb_classifier_multi_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier(objective="reg:logistic")
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, X.shape[1]]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(
            xgb,
            suffix="RegLog",
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )

    def test_xgb_classifier_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0
        xgb = XGBClassifier(objective="binary:logistic")
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, X.shape[1]]))],
            options={id(xgb): {"zipmap": False}},
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        self.assertTrue(conv_model is not None)
        sess = InferenceSession(
            conv_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": X.astype(np.float32)})
        assert_almost_equal(xgb.predict_proba(X), res[1])
        assert_almost_equal(xgb.predict(X), res[0])

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    def test_model_stacking_classifier_column_transformer(self):
        classifiers = {
            "A": XGBClassifier(n_estimators=5, random_state=42),
            "B": XGBClassifier(n_estimators=5, random_state=42),
        }
        model_to_test = Pipeline(
            steps=[
                (
                    "cbe",
                    ColumnTransformer(
                        [
                            ("norm1", Normalizer(norm="l1"), [0, 1]),
                            ("norm2", Normalizer(norm="l2"), [2, 3]),
                        ]
                    ),
                ),
                (
                    "sc",
                    StackingClassifier(
                        estimators=list(map(tuple, classifiers.items())),
                        stack_method="predict_proba",
                        passthrough=False,
                    ),
                ),
            ]
        )
        iris = load_iris()
        X = iris.data.astype(np.float32)
        y = (iris.target == 0).astype(np.int32)
        model_to_test.fit(X, y)
        model_onnx = convert_sklearn(
            model_to_test,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            options={"zipmap": False},
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": X.astype(np.float32)})
        assert_almost_equal(model_to_test.predict_proba(X), res[1])
        assert_almost_equal(model_to_test.predict(X), res[0])

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    def test_model_stacking_classifier_column_transformer_custom(self):
        classifiers = {
            "A": XGBClassifier(n_estimators=5, random_state=42),
            "B": XGBClassifier(n_estimators=5, random_state=42),
        }
        model_to_test = Pipeline(
            steps=[
                (
                    "cbe",
                    ColumnTransformer(
                        [
                            ("norm1", Normalizer(norm="l1"), [0, 1]),
                            ("norm2", Normalizer(norm="l2"), [2, 3]),
                        ]
                    ),
                ),
                (
                    "sc",
                    StackingClassifier(
                        estimators=list(map(tuple, classifiers.items())),
                        stack_method="predict_proba",
                        passthrough=False,
                    ),
                ),
            ]
        )
        iris = load_iris()
        X = iris.data.astype(np.float32)
        df = pandas.DataFrame(X)
        df.columns = ["A", "B", "C", "D"]
        X[:, 0] = X[:, 0].astype(np.int64).astype(X.dtype)
        df["A"] = df.A.astype(np.int64)
        df["B"] = df.B.astype(np.float32)
        df["C"] = df.C.astype(np.str_)
        y = (iris.target == 0).astype(np.int32)
        model_to_test.fit(df, y)
        model_onnx = convert_sklearn(
            model_to_test,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            options={"zipmap": False},
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": X.astype(np.float32)})
        assert_almost_equal(model_to_test.predict_proba(df), res[1])
        assert_almost_equal(model_to_test.predict(df), res[0])


if __name__ == "__main__":
    unittest.main()
