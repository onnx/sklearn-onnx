# SPDX-License-Identifier: Apache-2.0


from distutils.version import StrictVersion
import unittest
import numpy as np
import onnx
from onnxruntime import __version__ as ort_version
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Lasso, LassoLars, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    DoubleTensorType, FloatTensorType, Int64TensorType)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model, fit_classification_model,
    fit_clustering_model,
    fit_regression_model, TARGET_OPSET)


class TestSklearnGridSearchCVModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_binary_float(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 4)}]
        clf = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=100, solver='lbfgs',
                               multi_class='ovr'),
            tuned_parameters, cv=5)
        model, X = fit_classification_model(clf, n_classes=2)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSearchBinaryFloat-Dec4")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_multiclass_float(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 4)}]
        clf = GridSearchCV(
            SVC(random_state=42, probability=True, gamma='auto'),
            tuned_parameters, cv=5)
        model, X = fit_classification_model(clf, n_classes=5)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSearchMulticlassFloat")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_binary_int(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 4)}]
        clf = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=100, solver='lbfgs',
                               multi_class='ovr'),
            tuned_parameters, cv=5)
        model, X = fit_classification_model(clf, n_classes=2, is_int=True)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSearchBinaryInt-Dec4")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_multiclass_int(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 4)}]
        clf = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=100, solver='lbfgs',
                               multi_class='multinomial'),
            tuned_parameters, cv=5)
        model, X = fit_classification_model(clf, n_classes=4, is_int=True)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSearchMulticlassInt-Dec4")

    def test_grid_search_regression_int(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 4)}]
        clf = GridSearchCV(Lasso(max_iter=100),
                           tuned_parameters, cv=5)
        model, X = fit_regression_model(clf, is_int=True)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSerachRegressionInt-OneOffArray-Dec4")

    def test_grid_search_regressor_float(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 4)}]
        clf = GridSearchCV(LassoLars(max_iter=100),
                           tuned_parameters, cv=5)
        model, X = fit_regression_model(clf)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSearchRegressionFloat-OneOffArray-Dec4")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime %s" % '0.4.0')
    def test_grid_search_gaussian_regressor_float(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 4)}]
        clf = GridSearchCV(GaussianProcessRegressor(),
                           tuned_parameters, cv=5)
        model, X = fit_regression_model(clf)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSearchGaussianRegressionFloat"
                     "-OneOffArray-Dec4")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime %s" % '0.4.0')
    def test_grid_search_gaussian_regressor_double(self):
        tuned_parameters = [{'alpha': np.logspace(-4, -0.5, 4)}]
        clf = GridSearchCV(GaussianProcessRegressor(),
                           tuned_parameters, cv=3)
        model, X = fit_regression_model(clf)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float64), model, model_onnx,
            basename="SklearnGridSearchGaussianRegressionDouble"
                     "-OneOffArray-Dec4")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_binary_float_nozipmap(self):
        tuned_parameters = [{'C': np.logspace(-1, 0, 30)}]
        clf = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=100, solver='lbfgs',
                               multi_class='ovr'),
            tuned_parameters, cv=5)
        model, X = fit_classification_model(clf, n_classes=2)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(clf): {'zipmap': False, 'raw_scores': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        assert "zipmap" not in str(model_onnx).lower()
        assert '"LOGISTIC"' not in str(model_onnx).lower()
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnGridSearchBinaryFloat-Out0",
            methods=['predict', 'decision_function'])

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_svm(self):
        rand_seed = 0
        np.random.seed(rand_seed)

        def convert_to_onnx(sklearn_model, X, model_savename):
            onnx_model = to_onnx(sklearn_model, X[:1].astype(np.float32),
                                 target_opset=TARGET_OPSET)
            onnx.checker.check_model(onnx_model)
            return onnx_model

        def load_train_test():
            iris = load_iris()
            X = iris.data
            y = iris.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=rand_seed)
            return X_train, X_test, y_train, y_test

        def train_svc_gs(X_train, y_train, apply_fix=False):
            param_grid = {'C': [0.1, 1, 1e1], 'gamma': [1e-3, 1e-2, 1e-1]}
            clf_est = SVC(kernel='rbf', coef0=0.0, degree=3,
                          decision_function_shape='ovr',
                          probability=True)
            clf = GridSearchCV(clf_est, param_grid)
            clf.fit(X_train, y_train)
            return clf

        def run():
            # Load train and test dataset
            X_train, X_test, y_train, y_test = load_train_test()
            clf = train_svc_gs(X_train, y_train)
            onnx_model_name = "svc_gs_not_valid"
            return X_test, clf, convert_to_onnx(clf, X_test, onnx_model_name)

        x_test, model, model_onnx = run()
        dump_data_and_model(
            x_test.astype(np.float32), model, model_onnx,
            basename="SklearnGridSearchSVC-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_grid_search_binary_kmeans(self):
        tuned_parameters = [{'n_clusters': [2, 3]}]
        clf = GridSearchCV(KMeans(), tuned_parameters, cv=5)
        model, X = fit_clustering_model(clf, n_classes=2)
        X = X.astype(np.float32)
        model_onnx = convert_sklearn(
            model, "GridSearchCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model.best_estimator_, model_onnx,
            basename="SklearnGridSearchKMeans")


if __name__ == "__main__":
    unittest.main()
