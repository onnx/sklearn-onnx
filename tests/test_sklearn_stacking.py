# SPDX-License-Identifier: Apache-2.0

"""Tests StackingClassifier and StackingRegressor converter."""

import unittest
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
from onnx import TensorProto
import pandas
from onnxruntime import InferenceSession
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from sklearn.ensemble import StackingRegressor, StackingClassifier
except ImportError:
    # New in 0.22
    StackingRegressor = None
    StackingClassifier = None
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import (
    convert_sklearn,
    to_onnx,
    update_registered_converter,
    get_model_alias,
)
from skl2onnx.common.data_types import FloatTensorType
from test_utils import (
    dump_data_and_model,
    fit_regression_model,
    fit_classification_model,
    TARGET_OPSET,
)


def skl12():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("1.2")


def model_to_test_reg(passthrough=False):
    estimators = [("dt", DecisionTreeRegressor()), ("las", LinearRegression())]
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        passthrough=passthrough,
    )
    return stacking_regressor


def model_to_test_cl(passthrough=False):
    estimators = [("dt", DecisionTreeClassifier()), ("las", LogisticRegression())]
    stacking_regressor = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        passthrough=passthrough,
    )
    return stacking_regressor


class TestStackingConverter(unittest.TestCase):
    @unittest.skipIf(StackingRegressor is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_regression(self):
        model, X = fit_regression_model(model_to_test_reg())
        model_onnx = convert_sklearn(
            model,
            "stacking regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingRegressor-Dec4",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingRegressor is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_regression_passthrough(self):
        model, X = fit_regression_model(model_to_test_reg(passthrough=True), factor=0.1)
        model_onnx = convert_sklearn(
            model,
            "stacking regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingRegressorPassthrough",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_classifier(self):
        model, X = fit_classification_model(model_to_test_cl(), n_classes=2)
        model_onnx = convert_sklearn(
            model,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingClassifier",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_classifier_passthrough(self):
        model, X = fit_classification_model(
            model_to_test_cl(passthrough=True), n_classes=2
        )
        model_onnx = convert_sklearn(
            model,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingClassifierPassthrough",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_classifier_nozipmap(self):
        model, X = fit_classification_model(model_to_test_cl(), n_classes=2)
        model_onnx = convert_sklearn(
            model,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {"zipmap": False}},
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingClassifierNoZipMap",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_classifier_nozipmap_passthrough(self):
        model, X = fit_classification_model(
            model_to_test_cl(passthrough=True), n_classes=2
        )
        model_onnx = convert_sklearn(
            model,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {"zipmap": False}},
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingClassifierNoZipMapPassthrough",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    @unittest.skipIf(not skl12(), reason="sparse_output")
    def test_issue_786_exc(self):
        pipeline = make_pipeline(
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
                    (
                        "gb",
                        GradientBoostingClassifier(n_estimators=10, random_state=42),
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=2)),
                ],
                final_estimator=LogisticRegression(),
                cv=2,
            ),
        )

        X_train = pandas.DataFrame(
            dict(
                text=["A", "B", "A", "B", "AA", "B", "A", "B", "A", "AA", "B", "B"],
                val=[
                    0.5,
                    0.6,
                    0.7,
                    0.61,
                    0.51,
                    0.67,
                    0.51,
                    0.61,
                    0.71,
                    0.611,
                    0.511,
                    0.671,
                ],
            )
        )
        X_train["val"] = X_train.val.astype(numpy.float32)
        y_train = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pipeline.fit(X_train, y_train)
        with self.assertRaises(RuntimeError):
            to_onnx(pipeline, X=X_train[:1], target_opset=TARGET_OPSET)

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    @unittest.skipIf(not skl12(), reason="sparse_output")
    def test_issue_786(self):
        pipeline = make_pipeline(
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
                    (
                        "gb",
                        GradientBoostingClassifier(n_estimators=10, random_state=42),
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=2)),
                ],
                final_estimator=LogisticRegression(),
                cv=2,
            ),
        )

        X_train = pandas.DataFrame(
            dict(
                text=["A", "B", "A", "B", "AA", "B", "A", "B", "A", "AA", "B", "B"],
                val=[
                    0.5,
                    0.6,
                    0.7,
                    0.61,
                    0.51,
                    0.67,
                    0.51,
                    0.61,
                    0.71,
                    0.611,
                    0.511,
                    0.671,
                ],
            )
        )
        X_train["val"] = (X_train.val * 1000).astype(numpy.float32)
        y_train = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pipeline.fit(X_train, y_train)
        onx = to_onnx(
            pipeline,
            X=X_train[:1],
            options={"zipmap": False},
            target_opset=TARGET_OPSET,
        )
        # with open("ohe_debug.onnx", "wb") as f:
        #     f.write(onx.SerializeToString())
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(
            None,
            {
                "text": X_train.text.values.reshape((-1, 1)),
                "val": X_train.val.values.reshape((-1, 1)),
            },
        )
        assert_almost_equal(pipeline.predict(X_train), res[0])
        assert_almost_equal(pipeline.predict_proba(X_train), res[1])

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_classifier_column_transformer(self):
        classifiers = {
            "A": RandomForestClassifier(n_estimators=5, random_state=42),
            "B": GradientBoostingClassifier(n_estimators=5, random_state=42),
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
        model, X = fit_classification_model(model_to_test, n_classes=2)
        model_onnx = convert_sklearn(
            model,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingClassifierPipe",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_model_stacking_classifier_column_transformer_passthrough(self):
        classifiers = {
            "A": RandomForestClassifier(n_estimators=5, random_state=42),
            "B": GradientBoostingClassifier(n_estimators=5, random_state=42),
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
                        passthrough=True,
                    ),
                ),
            ]
        )
        model, X = fit_classification_model(model_to_test, n_classes=2)
        model_onnx = convert_sklearn(
            model,
            "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnStackingClassifierPipePassthrough",
            comparable_outputs=[0],
        )

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_concat_stacking(self):
        class CustomTransformer:
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        def shape_calculator(operator):
            pass

        def parser(scope, model, inputs, custom_parsers=None):
            alias = get_model_alias(type(model))
            op = scope.declare_local_operator(alias, model)
            op.inputs = inputs
            n_features = sum(list(map(lambda x: x.type.shape[1], op.inputs)))
            variable = scope.declare_local_variable(
                "c_outputs", FloatTensorType([None, n_features])
            )
            op.outputs.append(variable)
            return op.outputs

        def converter(scope, operator, container):
            output_cols = []

            for index in range(operator.inputs[0].type.shape[1]):
                index_name = scope.get_unique_variable_name("ind%d" % index)
                container.add_initializer(index_name, TensorProto.INT64, [], [index])
                feature_column_name = scope.get_unique_variable_name("fc%d" % index)
                container.add_node(
                    "ArrayFeatureExtractor",
                    [operator.inputs[0].full_name, index_name],
                    feature_column_name,
                    op_domain="ai.onnx.ml",
                    name=scope.get_unique_operator_name("AFE%d" % index),
                )
                output_cols.append(feature_column_name)

            container.add_node(
                "Concat",
                output_cols,
                operator.outputs[0].full_name,
                name=scope.get_unique_operator_name("CUSTOMCONCAT"),
                axis=-1,
            )

        update_registered_converter(
            CustomTransformer,
            "CustomTransformerUT",
            shape_calculator,
            converter,
            parser=parser,
            overwrite=True,
        )

        clf1 = RandomForestClassifier(n_estimators=5)
        clf2 = RandomForestClassifier(n_estimators=5)
        classifiers = {"clf1": clf1, "clf2": clf2}

        stacking_ensemble = StackingClassifier(
            estimators=list(map(tuple, classifiers.items())),
            n_jobs=1,
            stack_method="predict_proba",
            passthrough=False,
        )

        pipe = Pipeline(steps=[("ct", CustomTransformer()), ("sc", stacking_ensemble)])
        x = numpy.random.randn(20, 4).astype(numpy.float32)
        y = numpy.random.randint(2, size=20).astype(numpy.int64)
        pipe.fit(x, y)

        input_types = [("X", FloatTensorType([None, x.shape[1]]))]
        model_onnx = convert_sklearn(
            pipe,
            "bug",
            input_types,
            target_opset=TARGET_OPSET,
            verbose=0,
            options={"zipmap": False},
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(got.shape[0], x.shape[0])

    @unittest.skipIf(StackingClassifier is None, reason="new in 0.22")
    @ignore_warnings(category=FutureWarning)
    def test_concat_stacking_passthrough(self):
        class CustomTransformer:
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        def shape_calculator(operator):
            pass

        def parser(scope, model, inputs, custom_parsers=None):
            alias = get_model_alias(type(model))
            op = scope.declare_local_operator(alias, model)
            op.inputs = inputs
            n_features = sum(list(map(lambda x: x.type.shape[1], op.inputs)))
            variable = scope.declare_local_variable(
                "c_outputs", FloatTensorType([None, n_features])
            )
            op.outputs.append(variable)
            return op.outputs

        def converter(scope, operator, container):
            output_cols = []

            for index in range(operator.inputs[0].type.shape[1]):
                index_name = scope.get_unique_variable_name("ind%d" % index)
                container.add_initializer(index_name, TensorProto.INT64, [], [index])
                feature_column_name = scope.get_unique_variable_name("fc%d" % index)
                container.add_node(
                    "ArrayFeatureExtractor",
                    [operator.inputs[0].full_name, index_name],
                    feature_column_name,
                    op_domain="ai.onnx.ml",
                    name=scope.get_unique_operator_name("AFE%d" % index),
                )
                output_cols.append(feature_column_name)

            container.add_node(
                "Concat",
                output_cols,
                operator.outputs[0].full_name,
                name=scope.get_unique_operator_name("CUSTOMCONCAT"),
                axis=-1,
            )

        update_registered_converter(
            CustomTransformer,
            "CustomTransformerUT",
            shape_calculator,
            converter,
            parser=parser,
            overwrite=True,
        )

        clf1 = RandomForestClassifier(n_estimators=5)
        clf2 = RandomForestClassifier(n_estimators=5)
        classifiers = {"clf1": clf1, "clf2": clf2}

        stacking_ensemble = StackingClassifier(
            estimators=list(map(tuple, classifiers.items())),
            n_jobs=1,
            stack_method="predict_proba",
            passthrough=True,
        )

        pipe = Pipeline(steps=[("ct", CustomTransformer()), ("sc", stacking_ensemble)])
        x = numpy.random.randn(20, 4).astype(numpy.float32)
        y = numpy.random.randint(2, size=20).astype(numpy.int64)
        pipe.fit(x, y)

        input_types = [("X", FloatTensorType([None, x.shape[1]]))]
        model_onnx = convert_sklearn(
            pipe,
            "bug",
            input_types,
            target_opset=TARGET_OPSET,
            verbose=0,
            options={"zipmap": False},
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"X": x})[0]
        self.assertEqual(got.shape[0], x.shape[0])


if __name__ == "__main__":
    # import logging
    # log = logging.getLogger('skl2onnx')
    # log.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestStackingConverter().test_concat_stacking()
    unittest.main()
