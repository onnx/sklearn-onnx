# SPDX-License-Identifier: Apache-2.0

"""
Tests on functions in *onnx_helper*.
"""
import unittest
from distutils.version import StrictVersion
import numpy
import pandas
from onnxruntime import __version__ as ort_version
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
try:
    from sklearn.ensemble import VotingRegressor
except ImportError:
    # New in 0.21
    VotingRegressor = None
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # not available in 0.19
    ColumnTransformer = None
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from skl2onnx.common.utils_sklearn import enumerate_model_names
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    FloatTensorType, StringTensorType)
from skl2onnx.common.utils_sklearn import (
    _process_options, _process_pipeline_options)
from test_utils import (
    dump_data_and_model, fit_regression_model, TARGET_OPSET)


ort_version = ort_version.split('+')[0]


class TestUtilsSklearn(unittest.TestCase):

    @unittest.skipIf(VotingRegressor is None,
                     reason="new in 0.21")
    def test_voting_regression(self):
        model = VotingRegressor([
            ('lr', LinearRegression()),
            ('dt', DecisionTreeRegressor())])
        model, _ = fit_regression_model(model)
        names = list(enumerate_model_names(model))
        assert len(names) == 3
        assert [_[0] for _ in names] == ['', 'lr', 'dt']
        assert all(map(lambda x: isinstance(x, tuple), names))
        assert all(map(lambda x: len(x) == 2, names))

    def test_random_forest(self):
        model = RandomForestRegressor()
        model, _ = fit_regression_model(model)
        names = list(enumerate_model_names(model))
        assert all(map(lambda x: isinstance(x, tuple), names))
        assert all(map(lambda x: len(x) == 2, names))

    def test_pipeline(self):
        data = numpy.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=numpy.float32,
        )
        model = Pipeline([
            ("scaler1", StandardScaler()),
            (
                "union",
                FeatureUnion([
                    ("scaler2", StandardScaler()),
                    ("scaler3", MinMaxScaler()),
                ]),
            ),
        ])
        model.fit(data)
        names = list(enumerate_model_names(model))
        assert [_[0] for _ in names] == ['', 'scaler1', 'union',
                                         'union__scaler2', 'union__scaler3']

    def test_pipeline_lr(self):
        data = numpy.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=numpy.float32)
        yd = numpy.array([0, 1, 0, 2], dtype=numpy.float32)
        pipe = Pipeline([
            ('norm', MinMaxScaler()),
            ('clr', LogisticRegression())
        ])
        pipe.fit(data, yd)

        options = {'clr__raw_scores': True, 'clr__zipmap': False}
        new_options = _process_options(pipe, options)
        exp = {'raw_scores': True, 'zipmap': False}
        op = pipe.steps[1][1]
        self.assertIn(id(op), new_options)
        self.assertEqual(new_options[id(op)], exp)

        model_def = to_onnx(
            pipe, data,
            options={'clr__raw_scores': True, 'clr__zipmap': False},
            target_opset=TARGET_OPSET)
        sonx = str(model_def)
        assert "SOFTMAX" not in sonx

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime too old")
    def test_pipeline_column_transformer(self):

        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(
            lambda x: "cat1" if x > 0.5 else "cat2")
        X_train["vcat2"] = X_train["vB"].apply(
            lambda x: "cat3" if x > 0.5 else "cat4")
        y_train = y % 2
        numeric_features = [0, 1, 2]  # ["vA", "vB", "vC"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01,
            class_weight=dict(zip([False, True], [0.2, 0.8])),
            n_jobs=1, max_iter=10, solver="lbfgs", tol=1e-3)

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            (
                "onehot",
                OneHotEncoder(sparse=True, handle_unknown="ignore")),
            (
                "tsvd",
                TruncatedSVD(n_components=1, algorithm="arpack", tol=1e-4))])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)])

        model = Pipeline(steps=[("precprocessor",
                                 preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)
        names = list(enumerate_model_names(model, short=False))
        simple = [_[0] for _ in names]
        assert len(set(simple)) == len(simple)
        names = list(enumerate_model_names(model))
        simple2 = [_[0] for _ in names]
        assert len(simple2) == len(simple)
        exp = [
            '', 'precprocessor', 'precprocessor__num',
            'precprocessor__num__imputer', 'precprocessor__num__scaler',
            'precprocessor__cat', 'precprocessor__cat__onehot',
            'precprocessor__cat__onehot__categories___0',
            'precprocessor__cat__onehot__categories___1',
            'precprocessor__cat__tsvd', 'classifier']
        self.assertEqual(simple2[:len(exp) - 2], exp[:-2])

        initial_type = [
            ("numfeat", FloatTensorType([None, 3])),
            ("strfeat", StringTensorType([None, 2]))]
        model_onnx = convert_sklearn(model, initial_types=initial_type,
                                     target_opset=TARGET_OPSET)
        dump_data_and_model(
            X_train, model, model_onnx,
            basename="SklearnPipelineColumnTransformerPipelinerOptions1")

        options = {'classifier': {'zipmap': False}}
        new_options = _process_options(model, options)
        assert len(new_options) == 2

        model_onnx = convert_sklearn(
            model, initial_types=initial_type,
            options={'classifier': {'zipmap': False}},
            target_opset=TARGET_OPSET)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_train, model, model_onnx,
            basename="SklearnPipelineColumnTransformerPipelinerOptions2")

        options = {'classifier__zipmap': False}
        new_options = _process_options(model, options)
        assert len(new_options) == 2

        model_onnx = convert_sklearn(
            model, initial_types=initial_type,
            options=options, target_opset=TARGET_OPSET)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_train, model, model_onnx,
            basename="SklearnPipelineColumnTransformerPipelinerOptions2")

        options = {id(model): {'zipmap': False}}
        new_options = _process_pipeline_options(model, options)

        model_onnx = convert_sklearn(
            model, initial_types=initial_type,
            options={id(model): {'zipmap': False}},
            target_opset=TARGET_OPSET)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_train, model, model_onnx,
            basename="SklearnPipelineColumnTransformerPipelinerOptions2")


if __name__ == "__main__":
    unittest.main()
