import unittest
import numpy
from numpy.testing import assert_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import onnxruntime
from skl2onnx import convert_sklearn
from skl2onnx.helpers import collect_intermediate_steps, compare_objects
from skl2onnx.helpers import enumerate_pipeline_models
from skl2onnx.helpers.investigate import _alter_model_for_debugging
from skl2onnx.common import MissingShapeCalculator
from skl2onnx.common.data_types import (
    FloatTensorType,
)
from skl2onnx.common.data_types import guess_data_type


class MyScaler(StandardScaler):
    pass


class TestInvestigate(unittest.TestCase):

    def test_simple_pipeline(self):
        data = numpy.array([[0, 0], [0, 0], [2, 1], [2, 1]],
                           dtype=numpy.float32)
        model = Pipeline([("scaler1", StandardScaler()),
                          ("scaler2", StandardScaler())])
        model.fit(data)
        all_models = list(enumerate_pipeline_models(model))

        steps = collect_intermediate_steps(model, "pipeline",
                                           [("input",
                                             FloatTensorType([1, 2]))])

        assert len(steps) == 2
        assert len(all_models) == 3

        model.transform(data)
        for step in steps:
            short_onnx = step['short_onnx']
            sess = onnxruntime.InferenceSession(short_onnx.SerializeToString())
            onnx_outputs = sess.run(None, {'input': data})
            onnx_output = onnx_outputs[0]
            skl_outputs = step['model']._debug.outputs['transform']
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
            collect_intermediate_steps(model, "pipeline",
                                       [("input", FloatTensorType([1, 2]))])
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
                short_onnx = convert_sklearn(step_model, initial_types=t)
            except MissingShapeCalculator as e:
                if "MyScaler" in str(e):
                    continue
                raise
            sess = onnxruntime.InferenceSession(short_onnx.SerializeToString())
            onnx_outputs = sess.run(None, {'input': data_in})
            onnx_output = onnx_outputs[0]
            skl_outputs = step_model._debug.outputs['transform']
            assert_almost_equal(onnx_output, skl_outputs)
            compare_objects(onnx_output, skl_outputs)


if __name__ == "__main__":
    unittest.main()
