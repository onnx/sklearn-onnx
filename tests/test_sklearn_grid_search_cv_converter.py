# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
import unittest
import numpy as np
from onnxruntime import __version__ as ort_version
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Lasso, LassoLars, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_regression_model,
)


class TestSklearnGridSearchCVModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_binary_float(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 30)}]
        clf = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=100, solver='lbfgs',
                               multi_class='ovr'),
            tuned_parameters, cv=5, iid=False)
        model, X = fit_classification_model(clf, n_classes=2)
        model_onnx = convert_sklearn(
            model,
            "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGridSearchBinaryFloat-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_multiclass_float(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 30)}]
        clf = GridSearchCV(
            SVC(random_state=42, probability=True, gamma='auto'),
            tuned_parameters, cv=5, iid=False)
        model, X = fit_classification_model(clf, n_classes=5)
        model_onnx = convert_sklearn(
            model,
            "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGridSearchMulticlassFloat",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_binary_int(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 30)}]
        clf = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=100, solver='lbfgs',
                               multi_class='ovr'),
            tuned_parameters, cv=5, iid=False)
        model, X = fit_classification_model(clf, n_classes=2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "GridSearchCV",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGridSearchBinaryInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_multiclass_int(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 30)}]
        clf = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=100, solver='lbfgs',
                               multi_class='multinomial'),
            tuned_parameters, cv=5, iid=False)
        model, X = fit_classification_model(clf, n_classes=4, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "GridSearchCV",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGridSearchMulticlassInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_grid_search_regression_int(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 30)}]
        clf = GridSearchCV(Lasso(max_iter=100),
                           tuned_parameters, cv=5, iid=False)
        model, X = fit_regression_model(clf, is_int=True)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGridSerachRegressionInt-OneOffArray-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    def test_grid_search_regressor_float(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 30)}]
        clf = GridSearchCV(LassoLars(max_iter=100),
                           tuned_parameters, cv=5, iid=False)
        model, X = fit_regression_model(clf)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGridSearchRegressionFloat-OneOffArray-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime %s" % '0.4.0')
    def test_grid_search_gaussian_regressor_float(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 30)}]
        clf = GridSearchCV(GaussianProcessRegressor(),
                           tuned_parameters, cv=5, iid=False)
        model, X = fit_regression_model(clf)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGridSearchGaussianRegressionFloat"
                     "-OneOffArray-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.4.0') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime %s" % '0.4.0')
    def test_grid_search_gaussian_regressor_double(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 30)}]
        clf = GridSearchCV(GaussianProcessRegressor(),
                           tuned_parameters, cv=3, iid=False)
        model, X = fit_regression_model(clf)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            dtype=np.float64)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float64),
            model,
            model_onnx,
            basename="SklearnGridSearchGaussianRegressionDouble"
                     "-OneOffArray-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) "
            "<= StrictVersion('0.4.0') or "
            "StrictVersion(onnx.__version__) "
            "== StrictVersion('1.4.1')",
        )


if __name__ == "__main__":
    unittest.main()
