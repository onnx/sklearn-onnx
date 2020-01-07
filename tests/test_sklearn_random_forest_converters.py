# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from distutils.version import StrictVersion
import numpy
import sklearn
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from skl2onnx.common.data_types import onnx_built_with_ml, FloatTensorType
from skl2onnx import convert_sklearn
from test_utils import (
    convert_model,
    dump_one_class_classification,
    dump_binary_classification,
    dump_data_and_model,
    dump_multiple_classification,
    dump_multiple_regression,
    dump_single_regression,
    fit_classification_model,
    fit_multilabel_classification_model,
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


class TestSklearnTreeEnsembleModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=3)
        dump_one_class_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')"
                          " or StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_binary_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')"
                          " or StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')"
                          " or StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
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
                                             FloatTensorType([None, 2]))])
        dump_data_and_model(X, model, model_onnx,
                            basename=prefix + "Bin" +
                            model.__class__.__name__ +
                            '_mismatched_estimator_counts')

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
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
                                             FloatTensorType([None, 4]))])
        dump_data_and_model(X_test, clr, model_onnx,
                            basename=prefix + "RegMis" +
                            clr.__class__.__name__ +
                            '_mismatched_estimator_counts')

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_random_forest_regressor(self):
        model = RandomForestRegressor(n_estimators=3)
        dump_single_regression(
            model,
            allow_failure=("StrictVersion(onnxruntime.__version__)"
                           " <= StrictVersion('0.2.1')"),
        )
        dump_multiple_regression(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

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
                                             FloatTensorType([None, 2]))])
        dump_data_and_model(X, model, model_onnx,
                            basename=prefix + "Reg" +
                            model.__class__.__name__ +
                            "_mismatched_estimator_counts")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_extra_trees_classifier(self):
        model = ExtraTreesClassifier(n_estimators=3)
        dump_one_class_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_binary_classification(
            model,
            allow_failure=(
                "StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                "StrictVersion(onnxruntime.__version__)"
                " <= StrictVersion('0.2.1')"
            ),
        )
        dump_multiple_classification(
            model,
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure=(
                "StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                "StrictVersion(onnxruntime.__version__)"
                " <= StrictVersion('0.2.1')"
            ),
        )

    def test_extra_trees_regressor(self):
        model = ExtraTreesRegressor(n_estimators=3)
        dump_single_regression(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_multiple_regression(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multi_class_nocl(self):
        model, X = fit_classification_model(
            RandomForestClassifier(random_state=42),
            2, label_string=True)
        model_onnx = convert_sklearn(
            model,
            "multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}})
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X, model, model_onnx, classes=model.classes_,
            basename="SklearnRFMultiNoCl", verbose=False,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

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
            model, "unused", [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        X_test = X_test.astype(numpy.float32)[:5]
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnHGBRegressor", verbose=False,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="missing_go_to_left is missing")
    @unittest.skipIf(HistGradientBoostingRegressor is None,
                     reason="scikit-learn 0.22 + manual activation")
    def test_model_hgb_regressor_nonan(self):
        self.common_test_model_hgb_regressor(False)

    @unittest.skipIf(_sklearn_version() < StrictVersion('0.22.0'),
                     reason="NaN not allowed")
    @unittest.skipIf(HistGradientBoostingRegressor is None,
                     reason="scikit-learn 0.22 + manual activation")
    def test_model_hgb_regressor_nan(self):
        self.common_test_model_hgb_regressor(True)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_random_forest_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            RandomForestClassifier(random_state=42))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn RandomForestClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
        )
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnRandomForestClassifierMultiLabel-Out0",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_random_forest_classifier_multilabel_low_samples(self):
        model, X_test = fit_multilabel_classification_model(
            RandomForestClassifier(random_state=42), n_samples=4)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn RandomForestClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
        )
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnRandomForestClassifierMultiLabelLowSamples-Out0",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_extra_trees_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreesClassifier(random_state=42))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn ExtraTreesClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
        )
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnExtraTreesClassifierMultiLabel-Out0",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_extra_trees_classifier_multilabel_low_samples(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreesClassifier(random_state=42), n_samples=10)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn ExtraTreesClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
        )
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnExtraTreesClassifierMultiLabelLowSamples-Out0",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) <= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
