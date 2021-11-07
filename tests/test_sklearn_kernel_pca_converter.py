# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import numpy as np
from onnxruntime import __version__ as ort_version
from sklearn.datasets import load_diabetes
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET


ort_version = ".".join(ort_version.split('.')[:2])


class TestSklearnKernelPCAConverter(unittest.TestCase):

    def _fit_model(self, model, dtype=np.float32):
        data = load_diabetes()
        X_train, X_test, *_ = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42)
        model.fit(X_train)
        return model, X_test.astype(np.float32)

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="all needed operators not available")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.3.0'),
                     reason="discrepancies")
    @ignore_warnings(category=(FutureWarning, DeprecationWarning))
    def test_kernel_pca_default_float(self):
        model, X_test = self._fit_model(
            KernelPCA(random_state=42))
        model_onnx = to_onnx(model, X_test, target_opset=TARGET_OPSET)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnKernelPCA32")

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="all needed operators not available")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.3.0'),
                     reason="discrepancies")
    @ignore_warnings(category=(FutureWarning, DeprecationWarning))
    def test_kernel_pca_default_double(self):
        model, X_test = self._fit_model(
            KernelPCA(random_state=42, n_components=2), dtype=np.float64)
        model_onnx = to_onnx(model, X_test, target_opset=TARGET_OPSET)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnKernelPCA64")

    @unittest.skipIf(TARGET_OPSET < 13,
                     reason="all needed operators not available")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.3.0'),
                     reason="discrepancies")
    @ignore_warnings(category=(FutureWarning, DeprecationWarning))
    def test_kernel_pca_float(self):
        for kernel in ['rbf', 'cosine', 'sigmoid', 'poly', 'linear']:
            with self.subTest(kernel=kernel):
                model, X_test = self._fit_model(
                    KernelPCA(random_state=42, kernel=kernel,
                              n_components=4))
                model_onnx = to_onnx(model, X_test, target_opset=TARGET_OPSET)
                dump_data_and_model(
                    X_test, model, model_onnx,
                    basename="SklearnKernelPCA%s32" % kernel)

    @unittest.skipIf(TARGET_OPSET < 13,
                     reason="all needed operators not available")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.3.0'),
                     reason="discrepancies")
    @ignore_warnings(category=(FutureWarning, DeprecationWarning))
    def test_kernel_pca_double(self):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
            with self.subTest(kernel=kernel):
                model, X_test = self._fit_model(
                    KernelPCA(random_state=42, kernel=kernel,
                              n_components=4),
                    dtype=np.float64)
                model_onnx = to_onnx(model, X_test, target_opset=TARGET_OPSET)
                dump_data_and_model(
                    X_test, model, model_onnx,
                    basename="SklearnKernelPCA%s64" % kernel)

    @unittest.skipIf(TARGET_OPSET < 13,
                     reason="all needed operators not available")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.3.0'),
                     reason="discrepancies")
    @ignore_warnings(category=(FutureWarning, DeprecationWarning))
    def test_kernel_pca_double_cdist(self):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
            with self.subTest(kernel=kernel):
                model, X_test = self._fit_model(
                    KernelPCA(random_state=42, kernel=kernel,
                              n_components=4),
                    dtype=np.float64)
                model_onnx = to_onnx(model, X_test, target_opset=TARGET_OPSET,
                                     options={'optim': 'cdist'})
                dump_data_and_model(
                    X_test, model, model_onnx,
                    basename="SklearnKernelPCA%s64" % kernel)


if __name__ == "__main__":
    unittest.main()
