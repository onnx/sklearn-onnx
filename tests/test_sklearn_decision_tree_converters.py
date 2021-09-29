# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
from pandas import DataFrame
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    ExtraTreeClassifier, ExtraTreeRegressor
)
from sklearn.datasets import make_classification
from skl2onnx.common.data_types import onnx_built_with_ml
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx import convert_sklearn
from onnxruntime import InferenceSession, __version__ as ort_version
from test_utils import (
    binary_array_to_string,
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


ort_version = ort_version.split('+')[0]


class TestSklearnDecisionTreeModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.3.0"),
        reason="No suitable kernel definition found "
               "for op Cast(9) (node Cast)")
    def test_decisiontree_classifier1(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types,
                                     target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict_proba(X)
        if res[1][0][0] != pred[0, 0]:
            raise AssertionError("{}\n--\n{}".format(pred, DataFrame(res[1])))

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decisiontree_regressor0(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types,
                                     target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        if res[0][0, 0] != pred[0]:
            raise AssertionError("{}\n--\n{}".format(pred, DataFrame(res[1])))

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    def test_decisiontree_regressor_decision_path(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec.todense())
        assert exp == res[1].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    def test_decisiontree_regressor_decision_leaf(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp = path_to_leaf(model.tree_, dec.todense())
        assert exp.tolist() == res[1].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    def test_decisiontree_regressor_decision_path_leaf(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True,
                                 'decision_path': True}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp_leaf = path_to_leaf(model.tree_, dec.todense())
        exp_path = binary_array_to_string(dec.todense())
        assert exp_path == res[1].ravel().tolist()
        assert exp_leaf.tolist() == res[2].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    def test_decisiontree_classifier_decision_path(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True, 'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        prob = model.predict_proba(X)
        assert_almost_equal(prob, res[1])
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec.todense())
        assert exp == res[2].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    def test_decisiontree_classifier_decision_leaf(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True, 'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        prob = model.predict_proba(X)
        assert_almost_equal(prob, res[1])
        dec = model.decision_path(X)
        exp = path_to_leaf(model.tree_, dec.todense())
        assert exp.tolist() == res[2].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(TARGET_OPSET < 12, reason="LabelEncoder")
    def test_decisiontree_classifier_decision_path_leaf(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_leaf': True, 'decision_path': True,
                                 'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        prob = model.predict_proba(X)
        assert_almost_equal(prob, res[1])

        dec = model.decision_path(X)

        exp_path = binary_array_to_string(dec.todense())
        exp_leaf = path_to_leaf(model.tree_, dec.todense())
        assert exp_path == res[2].ravel().tolist()
        assert exp_leaf.tolist() == res[3].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decision_tree_classifier(self):
        model = DecisionTreeClassifier()
        dump_one_class_classification(model)
        dump_binary_classification(model)
        dump_multiple_classification(model)
        dump_multiple_classification(model, label_uint8=True)
        dump_multiple_classification(model, label_string=True)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_extra_tree_classifier(self):
        model = ExtraTreeClassifier()
        dump_one_class_classification(model)
        dump_binary_classification(model)
        dump_multiple_classification(model)

    def test_decision_tree_regressor(self):
        model = DecisionTreeRegressor()
        dump_single_regression(model)
        dump_multiple_regression(model)

    def test_extra_tree_regressor(self):
        model = ExtraTreeRegressor()
        dump_single_regression(model)
        dump_multiple_regression(model)

    def test_decision_tree_regressor_int(self):
        model, X = fit_regression_model(
            DecisionTreeRegressor(random_state=42), is_int=True)
        model_onnx = convert_sklearn(
            model, "decision tree regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnDecisionTreeRegressionInt")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multi_class_nocl(self):
        model, X = fit_classification_model(
            DecisionTreeClassifier(),
            4, label_string=True)
        model_onnx = convert_sklearn(
            model, "multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X, model, model_onnx, classes=model.classes_,
            basename="SklearnDTMultiNoCl")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_decision_tree_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            DecisionTreeClassifier(random_state=42))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn DecisionTreeClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnDecisionTreeClassifierMultiLabel-Out0")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_extra_tree_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreeClassifier(random_state=42))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn ExtraTreeClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnExtraTreeClassifierMultiLabel-Out0")

    def test_decision_tree_regressor_bool(self):
        model, X = fit_regression_model(
            DecisionTreeRegressor(random_state=42), is_bool=True)
        model_onnx = convert_sklearn(
            model, "decision tree regressor",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnDecisionTreeRegressionBool-Dec4")

    def test_extra_tree_regressor_bool(self):
        model, X = fit_regression_model(
            ExtraTreeRegressor(random_state=42), is_bool=True)
        model_onnx = convert_sklearn(
            model, "extra tree regressor",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnExtraTreeRegressionBool-Dec4")


if __name__ == "__main__":
    unittest.main()
