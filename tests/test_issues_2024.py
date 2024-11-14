# SPDX-License-Identifier: Apache-2.0
import unittest
import packaging.version as pv
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from onnxruntime import __version__ as ort_version


class TestInvestigate(unittest.TestCase):
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1053(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        import onnxruntime as rt
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import convert_sklearn

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Fitting logistic regression model.
        for cls in [LogisticRegression, DecisionTreeClassifier]:
            with self.subTest(cls=cls):
                clr = cls()  # Use logistic regression instead of decision tree.
                clr.fit(X_train, y_train)

                initial_type = [
                    ("float_input", FloatTensorType([None, 4]))
                ]  # Remove the batch dimension.
                onx = convert_sklearn(clr, initial_types=initial_type, target_opset=12)

                sess = rt.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                input_name = sess.get_inputs()[0].name
                label_name = sess.get_outputs()[0].name
                pred_onx = sess.run(
                    [label_name], {input_name: X_test[:1].astype("float32")}
                )[
                    0
                ]  # Select a single sample.
                self.assertEqual(len(pred_onx.tolist()), 1)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.16.0"),
        reason="opset 19 not implemented",
    )
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1055(self):
        import numpy as np
        from numpy.testing import assert_almost_equal
        import sklearn.feature_extraction.text
        import sklearn.linear_model
        import sklearn.pipeline
        import onnxruntime as rt
        import skl2onnx.common.data_types

        lr = sklearn.linear_model.LogisticRegression(
            C=100,
            multi_class="multinomial",
            solver="sag",
            class_weight="balanced",
            n_jobs=-1,
        )
        tf = sklearn.feature_extraction.text.TfidfVectorizer(
            token_pattern="\\w+|[^\\w\\s]+",
            ngram_range=(1, 1),
            max_df=1.0,
            min_df=1,
            sublinear_tf=True,
        )

        pipe = sklearn.pipeline.Pipeline([("transformer", tf), ("logreg", lr)])

        corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
            "more text",
            "$words",
            "I keep writing things",
            "how many documents now?",
            "this is a really long setence",
            "is this a final document?",
        ]
        labels = ["1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]

        pipe.fit(corpus, labels)

        onx = skl2onnx.convert_sklearn(
            pipe,
            "a model",
            initial_types=[
                ("input", skl2onnx.common.data_types.StringTensorType([None, 1]))
            ],
            target_opset=19,
            options={"zipmap": False},
        )
        for d in onx.opset_import:
            if d.domain == "":
                self.assertEqual(d.version, 19)
            elif d.domain == "com.microsoft":
                self.assertEqual(d.version, 1)
            elif d.domain == "ai.onnx.ml":
                self.assertEqual(d.version, 1)

        expected = pipe.predict_proba(corpus)
        sess = rt.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": np.array(corpus).reshape((-1, 1))})
        assert_almost_equal(expected, got[1], decimal=2)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.17.3"),
        reason="opset 19 not implemented",
    )
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1069(self):
        import math
        from typing import Any
        import numpy
        import pandas
        from sklearn import (
            base,
            compose,
            ensemble,
            linear_model,
            pipeline,
            preprocessing,
            datasets,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        import onnxruntime
        from skl2onnx import to_onnx
        from skl2onnx.sklapi import CastTransformer

        class FLAGS:
            classes = 7
            samples = 1000
            timesteps = 5
            trajectories = int(1000 / 5)
            features = 10
            seed = 10

        columns = [
            f"facelandmark{i}" for i in range(1, int(FLAGS.features / 2) + 1)
        ] + [f"poselandmark{i}" for i in range(1, int(FLAGS.features / 2) + 1)]

        X, y = datasets.make_classification(
            n_classes=FLAGS.classes,
            n_informative=math.ceil(math.log2(FLAGS.classes * 2)),
            n_samples=FLAGS.samples,
            n_features=FLAGS.features,
            random_state=FLAGS.seed,
        )

        X = pandas.DataFrame(X, columns=columns)

        X["trajectory"] = numpy.repeat(
            numpy.arange(FLAGS.trajectories), FLAGS.timesteps
        )
        X["timestep"] = numpy.tile(numpy.arange(FLAGS.timesteps), FLAGS.trajectories)

        trajectory_train, trajectory_test = train_test_split(
            X["trajectory"].unique(),
            test_size=0.25,
            random_state=FLAGS.seed,
        )

        trajectory_train, trajectory_test = set(trajectory_train), set(trajectory_test)

        X_train, X_test = (
            X[X["trajectory"].isin(trajectory_train)],
            X[X["trajectory"].isin(trajectory_test)],
        )
        y_train, _ = y[X_train.index], y[X_test.index]

        def augment_with_lag_timesteps(X, k, columns):
            augmented = X.copy()

            for i in range(1, k + 1):
                shifted = X[columns].groupby(X["trajectory"]).shift(i)
                shifted.columns = [f"{x}_lag{i}" for x in shifted.columns]

                augmented = pandas.concat([augmented, shifted], axis=1)

            return augmented

        X_train = augment_with_lag_timesteps(X_train, k=3, columns=X.columns[:-2])
        X_test = augment_with_lag_timesteps(X_test, k=3, columns=X.columns[:-2])

        X_train.drop(columns=["trajectory", "timestep"], inplace=True)
        X_test.drop(columns=["trajectory", "timestep"], inplace=True)

        def abc_Embedder() -> list[tuple[str, Any]]:
            return [
                ("cast64", CastTransformer(dtype=numpy.float64)),
                ("scaler", preprocessing.StandardScaler()),
                ("cast32", CastTransformer()),
                ("basemodel", DecisionTreeClassifier(max_depth=2)),
            ]

        def Classifier(features: list[str]) -> base.BaseEstimator:
            feats = [i for i, x in enumerate(features) if x.startswith("facelandmark")]

            classifier = ensemble.StackingClassifier(
                estimators=[
                    (
                        "facepipeline",
                        pipeline.Pipeline(
                            [
                                (
                                    "preprocessor",
                                    compose.ColumnTransformer(
                                        [("identity", "passthrough", feats)]
                                    ),
                                ),
                                ("embedder", pipeline.Pipeline(steps=abc_Embedder())),
                            ]
                        ),
                    ),
                    (
                        "posepipeline",
                        pipeline.Pipeline(
                            [
                                (
                                    "preprocessor",
                                    compose.ColumnTransformer(
                                        [("identity", "passthrough", feats)]
                                    ),
                                ),
                                ("embedder", pipeline.Pipeline(steps=abc_Embedder())),
                            ]
                        ),
                    ),
                ],
                final_estimator=linear_model.LogisticRegression(
                    multi_class="multinomial"
                ),
            )

            return classifier

        model = Classifier(list(X_train.columns))
        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            # If this fails, no need to go beyond.
            raise unittest.SkipTest(str(e))

        sample = X_train[:1].astype(numpy.float32)

        for m in [model.estimators_[0].steps[0][-1], model.estimators_[0], model]:
            with self.subTest(model=type(m)):
                exported = to_onnx(
                    model,
                    X=numpy.asarray(sample),
                    name="classifier",
                    target_opset={"": 12, "ai.onnx.ml": 2},
                    options={id(model): {"zipmap": False}},
                )

                modelengine = onnxruntime.InferenceSession(
                    exported.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                assert modelengine is not None

    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1046(self):
        import pandas as pd
        import numpy as np

        from xgboost import XGBClassifier

        from skl2onnx.common.data_types import (
            FloatTensorType,
            StringTensorType,
            Int64TensorType,
            DoubleTensorType,
        )
        from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )

        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == "int64":
                    t = Int64TensorType([None, 1])
                elif v == "float32":
                    t = FloatTensorType([None, 1])
                elif v == "float64":
                    t = DoubleTensorType([None, 1])
                else:
                    t = StringTensorType([None, 1])
                inputs.append((k, t))
            return inputs

        def get_categorical_features(df: pd.DataFrame):
            dtype_ser = df.dtypes
            categorical_features = dtype_ser[dtype_ser == "object"].index.tolist()
            return categorical_features

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        X_train = pd.DataFrame(
            [
                {"x1": 0.5, "c1": "A", "c2": "B"},
                {"x1": 0.6, "c1": "B", "c2": "B"},
                {"x1": 0.7, "c1": "A", "c2": "C"},
                {"x1": 0.8, "c1": "A", "c2": "B"},
                {"x1": 0.9, "c1": "B", "c2": "B"},
                {"x1": 1.1, "c1": "A", "c2": "C"},
                {"x1": 1.2, "c1": "B", "c2": "B"},
                {"x1": 1.3, "c1": "A", "c2": "B"},
            ]
        )
        y_train = (np.arange(X_train.shape[0]) % 3) % 2
        X_train["x1"] = X_train["x1"].astype(np.float32)

        categorical_features = get_categorical_features(X_train)
        numeric_features = [f for f in X_train.columns if not f in categorical_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        est_val = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        enable_categorical=False,
                        random_state=42,
                    ),
                ),
            ]
        )

        est_val.fit(X_train, y_train)

        # guess model input schema from training data
        schema = convert_dataframe_schema(X_train)

        # register XGBoost model converter
        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

        # convert sklearn pipeline to ONNX
        model_onnx = convert_sklearn(
            est_val,
            "xgb_pipeline",
            schema,
            target_opset={"": 18, "ai.onnx.ml": 3},
            options={"zipmap": False},
        )

        import onnxruntime

        sess = onnxruntime.InferenceSession(
            model_onnx.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )

        expected = est_val.predict_proba(X_train)
        feeds = {c: X_train[c].values.reshape((-1, 1)) for c in X_train.columns}
        got = sess.run(None, feeds)
        np.testing.assert_allclose(expected, got[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
