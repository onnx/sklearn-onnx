# SPDX-License-Identifier: Apache-2.0

"""
Tests on functions in *onnx_helper*.
"""
import unittest
from distutils.version import StrictVersion
import numpy
import onnx
from sklearn import __version__ as sklearn_version
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import (
    load_onnx_model,
    save_onnx_model,
    select_model_inputs_outputs,
    change_onnx_domain
)
from test_utils import TARGET_OPSET


def one_hot_encoder_supports_string():
    # StrictVersion does not work with development versions
    vers = '.'.join(sklearn_version.split('.')[:2])
    return StrictVersion(vers) >= StrictVersion("0.20.0")


class TestOnnxHelper(unittest.TestCase):
    def get_model(self, model):
        try:
            import onnxruntime  # noqa
        except ImportError:
            return None

        from onnxruntime import InferenceSession

        session = InferenceSession(save_onnx_model(model))
        return lambda X: session.run(None, {"input": X})[0]

    def test_onnx_helper_load_save(self):
        model = make_pipeline(StandardScaler(), Binarizer(threshold=0.5))
        X = numpy.array([[0.1, 1.1], [0.2, 2.2]])
        model.fit(X)
        model_onnx = convert_sklearn(model, "binarizer",
                                     [("input", FloatTensorType([None, 2]))],
                                     target_opset=TARGET_OPSET)
        filename = "temp_onnx_helper_load_save.onnx"
        save_onnx_model(model_onnx, filename)
        model = load_onnx_model(filename)
        new_model = select_model_inputs_outputs(model, "variable")
        assert new_model.graph is not None

        tr1 = self.get_model(model)
        tr2 = self.get_model(new_model)
        X = X.astype(numpy.float32)
        X1 = tr1(X)
        X2 = tr2(X)
        assert X1.shape == (2, 2)
        assert X2.shape == (2, 2)

    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20",
    )
    def test_onnx_helper_load_save_init(self):
        model = make_pipeline(
            Binarizer(),
            OneHotEncoder(sparse=False, handle_unknown='ignore'),
            StandardScaler())
        X = numpy.array([[0.1, 1.1], [0.2, 2.2], [0.4, 2.2], [0.2, 2.4]])
        model.fit(X)
        model_onnx = convert_sklearn(model, "pipe3",
                                     [("input", FloatTensorType([None, 2]))],
                                     target_opset=TARGET_OPSET)
        filename = "temp_onnx_helper_load_save.onnx"
        save_onnx_model(model_onnx, filename)
        model = load_onnx_model(filename)
        new_model = select_model_inputs_outputs(model, "variable")
        assert new_model.graph is not None

        tr1 = self.get_model(model)
        tr2 = self.get_model(new_model)
        X = X.astype(numpy.float32)
        X1 = tr1(X)
        X2 = tr2(X)
        assert X1.shape == (4, 2)
        assert X2.shape == (4, 2)

    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20",
    )
    def test_onnx_helper_load_save_init_meta(self):
        model = make_pipeline(Binarizer(), OneHotEncoder(sparse=False),
                              StandardScaler())
        X = numpy.array([[0.1, 1.1], [0.2, 2.2], [0.4, 2.2], [0.2, 2.4]])
        model.fit(X)
        model_onnx = convert_sklearn(model, "pipe3",
                                     [("input", FloatTensorType([None, 2]))],
                                     target_opset=TARGET_OPSET)
        meta = {'pA': 'one', 'pB': 'two'}
        onnx.helper.set_model_props(model_onnx, meta)
        new_model = select_model_inputs_outputs(model_onnx, "variable")
        vals = {p.key: p.value for p in new_model.metadata_props}
        assert vals == meta

    def test_change_onnx_domain(self):
        model = make_pipeline(StandardScaler())
        X = numpy.array([[0.1, 1.1], [0.2, 2.2], [0.4, 2.2], [0.2, 2.4]])
        model.fit(X)
        model_onnx = convert_sklearn(model, "pipe3",
                                     [("input", FloatTensorType([None, 2]))])
        model_onnx = change_onnx_domain(
            model_onnx, {'Scaler': ('ScalerNew', 'ML2')})
        self.assertIn('domain: "ML2"', str(model_onnx))
        self.assertIn('op_type: "ScalerNew"', str(model_onnx))


if __name__ == "__main__":
    unittest.main()
