# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import to_onnx
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model, fit_classification_model, TARGET_OPSET)


class TestSklearnGaussianProcessClassifier(unittest.TestCase):

    @unittest.skipIf(TARGET_OPSET < 12, reason="einsum")
    def test_gpc(self):

        gp = GaussianProcessClassifier()
        gp, X = fit_classification_model(gp, n_classes=2)

        # return_cov=False, return_std=False
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            dtype=np.float64, target_opset=TARGET_OPSET,
            options={GaussianProcessClassifier: {
                'zipmap': False, 'optim': 'cdist'}})
        self.assertTrue(model_onnx is not None)
        
        from mlprodict.onnxrt import OnnxInference
        oinf = OnnxInference(model_onnx)
        oinf.run({'X': X.astype(np.float64)}, verbose=1, fLOG=print)
        
        dump_data_and_model(
            X.astype(numpy.float64), gp, model_onnx,
            verbose=False, basename="SklearnGaussianProcessRBFT")


if __name__ == "__main__":
    unittest.main()
