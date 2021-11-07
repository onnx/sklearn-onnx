# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
import sklearn
from sklearn.datasets import (
    load_iris, make_regression, make_classification,
    load_boston
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
    onnx_built_with_ml,
)
from skl2onnx import convert_sklearn, to_onnx
from test_utils import (
    binary_array_to_string,
    convert_model,
    dump_one_class_classification,
    dump_binary_classification,
    dump_data_and_model,
    dump_multiple_classification,
    dump_multiple_regression,
    dump_single_regression,
    fit_classification_model,
    fit_multilabel_classification_model,
    fit_regression_model,
    path_to_leaf,
    TARGET_OPSET,
)
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor
    )
except ImportError:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None


def _sklearn_version():
    # Remove development version 0.22.dev0 becomes 0.22.
    v = ".".join(sklearn.__version__.split('.')[:2])
    return StrictVersion(v)


ort_version = ".".join(ort_version.split('.')[:2])


class TestSklearnTreeEnsembleModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=3)
        dump_one_class_classification(model)
        dump_binary_classification(model)
        dump_multiple_classification(model)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_random_forest_classifier_mismatched_estimator_counts(self):
        model = RandomForestClassifier(n_estimators=3)
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = ['A', 'B', 'A']
        model.fit(X, y)
        # Training code can manipulate n_estimators causing
        # n_estimators != len(estimators_). So simulate that here.
        model.n_estimators += 1
        model_onnx, prefix = convert_model(model, 'binary classifier',
                                           [('input',
                                             FloatTensorType([None, 2]))],
                                           target_opset=TARGET_OPSET)
        dump_data_and_model(X, model, model_onnx,
                            basename=prefix + "Bin" +
                            model.__class__.__name__ +
                            '_mismatched_estimator_counts')

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_random_forest_regressor_mismatches(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=13)
        X_test = X_test.astype(numpy.float32)
        clr = RandomForestRegressor(n_jobs=1, n_estimators=100)
        clr.fit(X_train, y_train)
        clr.fit(X, y)
        model_onnx, prefix = convert_model(clr, 'reg',
                                           [('input',
                                             FloatTensorType([None, 4]))],
                                           target_opset=TARGET_OPSET)
        dump_data_and_model(X_test, clr, model_onnx,
                            basename=prefix + "RegMis" +
                            clr.__class__.__name__ +
                            '_mismatched_estimator_counts')

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_random_forest_regressor(self):
        model = RandomForestRegressor(n_estimators=3)
        dump_single_regression(model)
        dump_multiple_regression(model)

    @ignore_warnings(category=FutureWarning)
    def test_random_forest_regressor_mismatched_estimator_counts(self):
        model = RandomForestRegressor(n_estimators=3)
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model.fit(X, y)
        # Training code can manipulate n_estimators causing
        # n_estimators != len(estimators_). So simulate that here.
        model.n_estimators += 1
        model_onnx, prefix = convert_model(model, 'single regressor',
                                           [('input',
                                             FloatTensorType([None, 2]))],
                                           target_opset=TARGET_OPSET)
        dump_data_and_model(X, model, model_onnx,
                            basename=prefix + "Reg" +
                            model.__class__.__name__ +
                            "_mismatched_estimator_counts")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_extra_trees_classifier(self):
        model = ExtraTreesClassifier(n_estimators=3)
        dump_one_class_classification(model)
        dump_binary_classification(model)
        dump_multiple_classification(model)

    @ignore_warnings(category=FutureWarning)
    def test_extra_trees_regressor(self):
        model = ExtraTreesRegressor(n_estimators=3)
        dump_single_regression(model)
        dump_multiple_regression(model)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_model_multi_class_nocl(self):
        model, X = fit_classification_model(
            RandomForestClassifier(random_state=42),
            2, label_string=True)
        model_onnx = convert_sklearn(
            model, "multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True, 'zipmap': False}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X[:5], model, model_onnx, classes=model.classes_,
            basename="SklearnRFMultiNoCl")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_model_multi_class_nocl_all(self):
        model, X = fit_classification_model(
            RandomForestClassifier(random_state=42),
            2, label_string=True)
        model_onnx = convert_sklearn(
            model, "multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True, 'zipmap': False}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        exp_label = model.predict(X)
        exp_proba = model.predict_proba(X)
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'input': X.astype(numpy.float32)})
        exp_label = numpy.array([int(cl[2:]) for cl in exp_label])
        assert_almost_equal(exp_proba, got[1], decimal=5)
        diff = numpy.abs(exp_label - got[0]).sum()
        if diff >= 3:
            # Both scikit-learn and onnxruntime do the computation
            # by parallelizing by trees. However, scikit-learn
            # always adds tree outputs in the same order,
            # onnxruntime does not. It may lead to small discrepencies.
            # This test ensures that probabilities are almost the same.
            # But a discrepencies around 0.5 may change the label.
            # That explains why the test allows less than 3 differences.
            assert_almost_equal(exp_label, got[0])

    @ignore_warnings(category=FutureWarning)
    def test_random_forest_classifier_int(self):
        model, X = fit_classification_model(
            RandomForestClassifier(n_estimators=5, random_state=42),
            3, is_int=True)
        model_onnx = convert_sklearn(
            model, "random forest classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRandomForestClassifierInt")

    @ignore_warnings(category=FutureWarning)
    def test_extra_trees_classifier_int(self):
        model, X = fit_classification_model(
            ExtraTreesClassifier(n_estimators=5, random_state=42),
            4, is_int=True)
        model_onnx = convert_sklearn(
            model, "extra trees classifier",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnExtraTreesClassifierInt")

    @ignore_warnings(category=FutureWarning)
    def test_random_forest_classifier_bool(self):
        model, X = fit_classification_model(
            RandomForestClassifier(n_estimators=5, random_state=42),
            3, is_bool=True)
        model_onnx = convert_sklearn(
            model, "random forest classifier",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRandomForestClassifierBool")

    @ignore_warnings(category=FutureWarning)
    def test_extra_trees_classifier_bool(self):
        model, X = fit_classification_model(
            ExtraTreesClassifier(n_estimators=5, random_state=42),
            2, is_bool=True)
        model_onnx = convert_sklearn(
            model, "extra trees regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnExtraTreesClassifierBool")

    @ignore_warnings(category=FutureWarning)
    def common_test_model_hgb_regressor(self, add_nan=False):
        model = HistGradientBoostingRegressor(max_iter=5, max_depth=2)
        X, y = make_regression(n_features=10, n_samples=1000,
                               n_targets=1, random_state=42)
        if add_nan:
            rows = numpy.random.randint(0, X.shape[0] - 1, X.shape[0] // 3)
            cols = numpy.random.randint(0, X.shape[1] - 1, X.shape[0] // 3)
            X[rows, cols] = numpy.nan

        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                       random_state=42)
        model.fit(X_train, y_train)

        model_onnx = convert_sklearn(
            model, "unused", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        X_test = X_test.astype(numpy.float32)[:5]
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnHGBRegressor", verbose=False)

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="missing_go_to_left is missing")
    @unittest.skipIf(HistGradientBoostingRegressor is None,
                     reason="scikit-learn 0.22 + manual activation")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.2.0'),
                     reason="issue with nan for earlier ort")
    @ignore_warnings(category=FutureWarning)
    def test_model_hgb_regressor_nonan(self):
        self.common_test_model_hgb_regressor(False)

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="NaN not allowed")
    @unittest.skipIf(HistGradientBoostingRegressor is None,
                     reason="scikit-learn 0.22 + manual activation")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.2.0'),
                     reason="issue with nan for earlier ort")
    @ignore_warnings(category=FutureWarning)
    def test_model_hgb_regressor_nan(self):
        self.common_test_model_hgb_regressor(True)

    def common_test_model_hgb_classifier(self, add_nan=False, n_classes=2):
        model = HistGradientBoostingClassifier(max_iter=5, max_depth=2)
        X, y = make_classification(n_features=10, n_samples=1000,
                                   n_informative=4, n_classes=n_classes,
                                   random_state=42)
        if add_nan:
            rows = numpy.random.randint(0, X.shape[0] - 1, X.shape[0] // 3)
            cols = numpy.random.randint(0, X.shape[1] - 1, X.shape[0] // 3)
            X[rows, cols] = numpy.nan

        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                       random_state=42)
        model.fit(X_train, y_train)

        model_onnx = convert_sklearn(
            model, "unused", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        X_test = X_test.astype(numpy.float32)[:5]

        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnHGBClassifier%s%d" % (
                "nan" if add_nan else '', n_classes),
            verbose=False)

        if n_classes == 2:
            model_onnx = convert_sklearn(
                model, "unused",
                [("input", FloatTensorType([None, X.shape[1]]))],
                options={model.__class__: {'raw_scores': True}},
                target_opset=TARGET_OPSET)
            self.assertIsNotNone(model_onnx)
            X_test = X_test.astype(numpy.float32)[:5]

            # There is a bug in onnxruntime <= 1.1.0.
            # Raw scores are always positive.
            dump_data_and_model(
                X_test, model, model_onnx,
                basename="SklearnHGBClassifierRaw%s%d" % (
                    "nan" if add_nan else '', n_classes),
                verbose=False,
                methods=['predict', 'decision_function_binary'])

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="missing_go_to_left is missing")
    @unittest.skipIf(HistGradientBoostingClassifier is None,
                     reason="scikit-learn 0.22 + manual activation")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.2.0'),
                     reason="issue with nan for earlier ort")
    @ignore_warnings(category=FutureWarning)
    def test_model_hgb_classifier_nonan(self):
        self.common_test_model_hgb_classifier(False)

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="NaN not allowed")
    @unittest.skipIf(HistGradientBoostingClassifier is None,
                     reason="scikit-learn 0.22 + manual activation")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.2.0'),
                     reason="issue with nan for earlier ort")
    @ignore_warnings(category=FutureWarning)
    def test_model_hgb_classifier_nan(self):
        self.common_test_model_hgb_classifier(True)

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="missing_go_to_left is missing")
    @unittest.skipIf(HistGradientBoostingClassifier is None,
                     reason="scikit-learn 0.22 + manual activation")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.2.0'),
                     reason="issue with nan for earlier ort")
    @ignore_warnings(category=FutureWarning)
    def test_model_hgb_classifier_nonan_multi(self):
        self.common_test_model_hgb_classifier(False, n_classes=3)

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="NaN not allowed")
    @unittest.skipIf(HistGradientBoostingClassifier is None,
                     reason="scikit-learn 0.22 + manual activation")
    @ignore_warnings(category=FutureWarning)
    def test_model_hgb_classifier_nan_multi(self):
        self.common_test_model_hgb_classifier(True, n_classes=3)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_model_random_forest_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            RandomForestClassifier(random_state=42, n_estimators=5))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn RandomForestClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnRandomForestClassifierMultiLabel-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_model_random_forest_classifier_multilabel_low_samples(self):
        model, X_test = fit_multilabel_classification_model(
            RandomForestClassifier(random_state=42, n_estimators=5),
            n_samples=4)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn RandomForestClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnRandomForestClassifierMultiLabelLowSamples-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_model_extra_trees_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreesClassifier(random_state=42, n_estimators=5))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn ExtraTreesClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnExtraTreesClassifierMultiLabel-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_model_extra_trees_classifier_multilabel_low_samples(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreesClassifier(random_state=42, n_estimators=5),
            n_samples=10)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn ExtraTreesClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnExtraTreesClassifierMultiLabelLowSamples-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_boston_pca_rf(self):
        data = load_boston()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0)
        pipe = Pipeline([
            ('acp', PCA(n_components=3)),
            ('rf', RandomForestRegressor())])
        pipe.fit(X_train, y_train)
        X32 = X_test.astype(numpy.float32)
        model_onnx = to_onnx(pipe, X32[:1], target_opset=TARGET_OPSET)
        dump_data_and_model(
            X32, pipe, model_onnx, methods=['predict'],
            basename="SklearnBostonPCARF-Dec4")

    @ignore_warnings(category=FutureWarning)
    def test_random_forest_regressor_int(self):
        model, X = fit_regression_model(
            RandomForestRegressor(n_estimators=5, random_state=42),
            is_int=True)
        model_onnx = convert_sklearn(
            model, "random forest regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRandomForestRegressorInt-Dec4",)

    @ignore_warnings(category=FutureWarning)
    def test_extra_trees_regressor_int(self):
        model, X = fit_regression_model(
            ExtraTreesRegressor(n_estimators=5, random_state=42),
            is_int=True)
        model_onnx = convert_sklearn(
            model, "extra trees regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnExtraTreesRegressorInt-Dec4")

    @ignore_warnings(category=FutureWarning)
    def test_random_forest_regressor_bool(self):
        model, X = fit_regression_model(
            RandomForestRegressor(n_estimators=5, random_state=42),
            is_bool=True)
        model_onnx = convert_sklearn(
            model, "random forest regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnRandomForestRegressorBool-Dec4")

    @ignore_warnings(category=FutureWarning)
    def test_extra_trees_regressor_bool(self):
        model, X = fit_regression_model(
            ExtraTreesRegressor(n_estimators=5, random_state=42),
            is_bool=True)
        model_onnx = convert_sklearn(
            model, "extra trees regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnExtraTreesRegressorBool-Dec4")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_randomforestregressor_decision_path(self):
        model = RandomForestRegressor(max_depth=2, n_estimators=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec[0].todense())
        got = numpy.array([''.join(row) for row in res[1]])
        assert exp == got.ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_extratreesregressor_decision_path(self):
        model = ExtraTreesRegressor(max_depth=2, n_estimators=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec[0].todense())
        got = numpy.array([''.join(row) for row in res[1]])
        assert exp == got.ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_randomforestclassifier_decision_path(self):
        model = RandomForestClassifier(max_depth=2, n_estimators=2)
        X, y = make_classification(3, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True, 'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        prob = model.predict_proba(X)
        assert_almost_equal(prob, res[1])
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec[0].todense())
        got = numpy.array([''.join(row) for row in res[2]])
        assert exp == got.ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_extratreesclassifier_decision_path(self):
        model = ExtraTreesClassifier(max_depth=2, n_estimators=3)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True, 'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        prob = model.predict_proba(X)
        assert_almost_equal(prob, res[1])
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec[0].todense())
        got = numpy.array([''.join(row) for row in res[2]])
        assert exp == got.ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_rf_regressor_decision_leaf(self):
        model = RandomForestRegressor(n_estimators=2, max_depth=3)
        X, y = make_regression(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True}},
            target_opset=TARGET_OPSET)

        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel(), decimal=4)
        dec = model.decision_path(X)
        exp = path_to_leaf(model.estimators_, dec[0].todense(), dec[1])
        assert exp.tolist() == res[1].tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_rf_regressor_decision_path_leaf(self):
        model = RandomForestRegressor(n_estimators=3, max_depth=3)
        X, y = make_regression(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True,
                                 'decision_path': True}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel(), decimal=4)
        dec = model.decision_path(X)
        exp_leaf = path_to_leaf(model.estimators_, dec[0].todense(), dec[1])
        exp_path = binary_array_to_string(dec[0].todense())
        got_path = numpy.array([''.join(row) for row in res[1]])
        assert exp_path == got_path.ravel().tolist()
        assert exp_leaf.tolist() == res[2].tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_rf_classifier_decision_leaf(self):
        model = RandomForestClassifier(n_estimators=2, max_depth=3)
        X, y = make_classification(3, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True, 'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp = path_to_leaf(model.estimators_, dec[0].todense(), dec[1])
        assert exp.tolist() == res[2].tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    @ignore_warnings(category=FutureWarning)
    def test_rf_classifier_decision_path_leaf(self):
        model = RandomForestClassifier(n_estimators=3, max_depth=3)
        X, y = make_classification(3, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True,
                                 'decision_path': True,
                                 'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(numpy.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp_leaf = path_to_leaf(model.estimators_, dec[0].todense(), dec[1])
        exp_path = binary_array_to_string(dec[0].todense())
        got_path = numpy.array([''.join(row) for row in res[2]])
        assert exp_path == got_path.ravel().tolist()
        assert exp_leaf.tolist() == res[3].tolist()


if __name__ == "__main__":
    unittest.main()
