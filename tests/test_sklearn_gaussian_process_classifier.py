# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import inspect
from io import StringIO
from distutils.version import StrictVersion
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    Sum, DotProduct, ExpSineSquared, RationalQuadratic,
    RBF, ConstantKernel as C,
)
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import to_onnx
from skl2onnx.proto import get_latest_tested_opset_version
from skl2onnx.operator_converters.gaussian_process import (
    convert_kernel, convert_kernel_diag
)
from onnxruntime import InferenceSession
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model, fit_classification_model, TARGET_OPSET)


class TestSklearnGaussianProcessClassifier(unittest.TestCase):

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="onnxruntime too old")
    def test_gpr_rbf_fitted_true(self):

        gp = GaussianProcessClassifier()
        gp, X = fit_classification_model(gp, n_classes=2)

        # return_cov=False, return_std=False
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            dtype=np.float64, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, gp, model_onnx,
                            verbose=False,
                            basename="SklearnGaussianProcessRBFT")


if __name__ == "__main__":
    unittest.main()
