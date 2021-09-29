# SPDX-License-Identifier: Apache-2.0

import unittest
import io
from contextlib import redirect_stdout
import numpy
from numpy.testing import assert_almost_equal
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # not avaiable in 0.19
    ColumnTransformer = None
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
import onnxruntime
from skl2onnx import convert_sklearn
from skl2onnx.helpers import (
    collect_intermediate_steps, compare_objects,
    enumerate_pipeline_models)
from skl2onnx.helpers.investigate import _alter_model_for_debugging
from skl2onnx.common import MissingShapeCalculator
from skl2onnx.common.data_types import (
    FloatTensorType, guess_data_type)
from test_utils import TARGET_OPSET


class MyScaler(StandardScaler):
    pass


class TestInvestigate(unittest.TestCase):

    def test_simple_pipeline(self):
        for opset in (11, TARGET_OPSET):
            if opset > TARGET_OPSET:
                continue
            data = numpy.array([[0, 0], [0, 0], [2, 1], [2, 1]],
                               dtype=numpy.float32)
            model = Pipeline([("scaler1", StandardScaler()),
                              ("scaler2", StandardScaler())])
            model.fit(data)
            all_models = list(enumerate_pipeline_models(model))

            steps = collect_intermediate_steps(
                model, "pipeline", [("input", FloatTensorType([None, 2]))],
                target_opset=opset)

            assert len(steps) == 2
            assert len(all_models) == 3

            expected = 'version:%d}' % opset
            expected1 = 'version:1}'
            model.transform(data)
            for step in steps:
                onnx_step = step['onnx_step']
                text = str(onnx_step).replace('\n', ' ').replace(' ', '')
                if expected not in text and expected1 not in text:
                    raise AssertionError(
                        "Unable to find '{}'\n'{}'\n".format(
                            expected, text))
                sess = onnxruntime.InferenceSession(
                    onnx_step.SerializeToString())
                onnx_outputs = sess.run(None, {'input': data})
                onnx_output = onnx_outputs[0]
                skl_outputs = step['model']._debug.outputs['transform']
                assert str(step['model']._debug) is not None
                sdt = step['model']._debug.display(data, 5)
                assert 'shape' in sdt
                assert_almost_equal(onnx_output, skl_outputs)
                compare_objects(onnx_output, skl_outputs)

    def test_missing_converter(self):
        data = numpy.array([[0, 0], [0, 0], [2, 1], [2, 1]],
                           dtype=numpy.float32)
        model = Pipeline([("scaler1", StandardScaler()),
                          ("scaler2", StandardScaler()),
                          ("scaler3", MyScaler())])
        model.fit(data)
        all_models = list(enumerate_pipeline_models(model))

        try:
            collect_intermediate_steps(
                model, "pipeline",
                [("input", FloatTensorType([None, 2]))],
                target_opset=TARGET_OPSET)
        except MissingShapeCalculator as e:
            assert "MyScaler" in str(e)
            assert "gallery" in str(e)

        _alter_model_for_debugging(model, recursive=True)
        model.transform(data)
        all_models = list(enumerate_pipeline_models(model))

        for ind, step, last in all_models:
            if ind == (0,):
                # whole pipeline
                continue
            step_model = step
            data_in = step_model._debug.inputs['transform']
            t = guess_data_type(data_in)
            try:
                onnx_step = convert_sklearn(step_model, initial_types=t,
                                            target_opset=TARGET_OPSET)
            except MissingShapeCalculator as e:
                if "MyScaler" in str(e):
                    continue
                raise
            sess = onnxruntime.InferenceSession(onnx_step.SerializeToString())
            onnx_outputs = sess.run(None, {'input': data_in})
            onnx_output = onnx_outputs[0]
            skl_outputs = step_model._debug.outputs['transform']
            assert_almost_equal(onnx_output, skl_outputs)
            compare_objects(onnx_output, skl_outputs)

    def test_simple_column_transformer(self):
        if ColumnTransformer is None:
            return
        data = numpy.array([[0, 0], [0, 0], [2, 1], [2, 1]],
                           dtype=numpy.float32)
        model = ColumnTransformer([("scaler1", StandardScaler(), [0]),
                                  ("scaler2", RobustScaler(), [1])])
        model.fit(data)
        all_models = list(enumerate_pipeline_models(model))

        steps = collect_intermediate_steps(
            model, "coulmn transformer",
            [("input", FloatTensorType([None, 2]))],
            target_opset=TARGET_OPSET)

        assert len(steps) == 2
        assert len(all_models) == 3

        model.transform(data)
        for step in steps:
            onnx_step = step['onnx_step']
            sess = onnxruntime.InferenceSession(onnx_step.SerializeToString())
            onnx_outputs = sess.run(None, {'input': data})
            onnx_output = onnx_outputs[0]
            skl_outputs = step['model']._debug.outputs['transform']
            assert_almost_equal(onnx_output, skl_outputs)
            compare_objects(onnx_output.tolist(), skl_outputs.tolist())

    def test_simple_feature_union(self):
        data = numpy.array([[0, 0], [0, 0], [2, 1], [2, 1]],
                           dtype=numpy.float32)
        model = FeatureUnion([("scaler1", StandardScaler()),
                             ("scaler2", RobustScaler())])
        model.fit(data)
        all_models = list(enumerate_pipeline_models(model))
        steps = collect_intermediate_steps(
            model, "feature union",
            [("input", FloatTensorType([None, 2]))],
            target_opset=TARGET_OPSET)

        assert len(steps) == 2
        assert len(all_models) == 3

        model.transform(data)
        for step in steps:
            onnx_step = step['onnx_step']
            sess = onnxruntime.InferenceSession(onnx_step.SerializeToString())
            onnx_outputs = sess.run(None, {'input': data})
            onnx_output = onnx_outputs[0]
            skl_outputs = step['model']._debug.outputs['transform']
            assert_almost_equal(onnx_output, skl_outputs)
            compare_objects(onnx_output, skl_outputs)

    def test_simple_pipeline_predict(self):
        data = load_iris()
        X, y = data.data, data.target
        model = Pipeline([("scaler1", StandardScaler()),
                          ("lr", LogisticRegression())])
        model.fit(X, y)
        all_models = list(enumerate_pipeline_models(model))

        steps = collect_intermediate_steps(
            model, "pipeline",
            [("input", FloatTensorType((None, X.shape[1])))],
            target_opset=TARGET_OPSET)

        assert len(steps) == 2
        assert len(all_models) == 3

        model.predict(X)
        for step in steps:
            onnx_step = step['onnx_step']
            sess = onnxruntime.InferenceSession(onnx_step.SerializeToString())
            onnx_outputs = sess.run(None, {'input': X.astype(numpy.float32)})
            onnx_output = onnx_outputs[0]
            dbg_outputs = step['model']._debug.outputs
            skl_outputs = (dbg_outputs['transform'] if 'transform' in
                           dbg_outputs else dbg_outputs['predict'])
            assert_almost_equal(onnx_output, skl_outputs, decimal=6)
            compare_objects(onnx_output, skl_outputs)

    def test_simple_pipeline_predict_proba(self):
        data = load_iris()
        X, y = data.data, data.target
        model = Pipeline([("scaler1", StandardScaler()),
                          ("lr", LogisticRegression())])
        model.fit(X, y)
        all_models = list(enumerate_pipeline_models(model))

        steps = collect_intermediate_steps(
            model, "pipeline",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        assert len(steps) == 2
        assert len(all_models) == 3

        model.predict_proba(X)
        for step in steps:
            onnx_step = step['onnx_step']
            sess = onnxruntime.InferenceSession(onnx_step.SerializeToString())
            onnx_outputs = sess.run(None, {'input': X.astype(numpy.float32)})
            dbg_outputs = step['model']._debug.outputs
            if 'transform' in dbg_outputs:
                onnx_output = onnx_outputs[0]
                skl_outputs = dbg_outputs['transform']
            else:
                onnx_output = onnx_outputs[1]
                skl_outputs = dbg_outputs['predict_proba']
            assert_almost_equal(onnx_output, skl_outputs, decimal=6)
            compare_objects(onnx_output, skl_outputs)

    def test_verbose(self):
        data = load_iris()
        X, y = data.data, data.target
        model = Pipeline([("scaler1", StandardScaler()),
                          ("lr", LogisticRegression())])
        model.fit(X, y)
        st = io.StringIO()
        with redirect_stdout(st):
            convert_sklearn(
                model, initial_types=[('X', FloatTensorType())],
                verbose=1)
        self.assertIn("[convert_sklearn] convert_topology", st.getvalue())


if __name__ == "__main__":
    unittest.main()
