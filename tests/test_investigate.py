import unittest
import numpy
from numpy.testing import assert_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import onnxruntime
from skl2onnx.helpers.intermediate import collect_intermediate_steps
from skl2onnx.common.data_types import (
    FloatTensorType,
)


class TestInvestigate(unittest.TestCase):

    def test_simple_pipeline(self):
        data = numpy.array([[0, 0], [0, 0], [2, 1], [2, 1]],
                           dtype=numpy.float32)
        model = Pipeline([("scaler1", StandardScaler()),
                          ("scaler2", StandardScaler())])
        model.fit(data)

        steps = collect_intermediate_steps(model, "pipeline",
                                           [("input",
                                             FloatTensorType([1, 2]))])

        assert len(steps) == 2

        model.transform(data)
        for step in steps:
            short_onnx = step['short_onnx']
            sess = onnxruntime.InferenceSession(short_onnx.SerializeToString())
            onnx_outputs = sess.run(None, {'input': data})
            onnx_output = onnx_outputs[0]
            skl_outputs = step['model']._debug.outputs['transform']
            assert_almost_equal(onnx_output, skl_outputs)


if __name__ == "__main__":
    unittest.main()
