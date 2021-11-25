# SPDX-License-Identifier: Apache-2.0

"""Tests StackingClassifier and StackingRegressor converter."""

import unittest
import numpy
from numpy.testing import assert_almost_equal
import pandas
from onnxruntime import InferenceSession
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier)
try:
    from sklearn.ensemble import StackingRegressor, StackingClassifier
except ImportError:
    # New in 0.22
    StackingRegressor = None
    StackingClassifier = None
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from test_utils import (
    dump_data_and_model, fit_regression_model,
    fit_classification_model, TARGET_OPSET)


def model_to_test_reg():
    estimators = [
        ('dt', DecisionTreeRegressor()),
        ('las', LinearRegression())]
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression())
    return stacking_regressor


def model_to_test_cl():
    estimators = [
        ('dt', DecisionTreeClassifier()),
        ('las', LogisticRegression())]
    stacking_regressor = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression())
    return stacking_regressor


class TestStackingConverter(unittest.TestCase):

    @unittest.skipIf(StackingRegressor is None,
                     reason="new in 0.22")
    def test_model_stacking_regression(self):
        model, X = fit_regression_model(model_to_test_reg())
        model_onnx = convert_sklearn(
            model, "stacking regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnStackingRegressor-Dec4",
            comparable_outputs=[0])

    @unittest.skipIf(StackingClassifier is None,
                     reason="new in 0.22")
    def test_model_stacking_classifier(self):
        model, X = fit_classification_model(
            model_to_test_cl(), n_classes=2)
        model_onnx = convert_sklearn(
            model, "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnStackingClassifier",
            comparable_outputs=[0])

    @unittest.skipIf(StackingClassifier is None,
                     reason="new in 0.22")
    def test_model_stacking_classifier_nozipmap(self):
        model, X = fit_classification_model(
            model_to_test_cl(), n_classes=2)
        model_onnx = convert_sklearn(
            model, "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'zipmap': False}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnStackingClassifierNoZipMap",
            comparable_outputs=[0])

    @unittest.skipIf(StackingClassifier is None,
                     reason="new in 0.22")
    def test_issue_786_exc(self):
        pipeline = make_pipeline(
            OneHotEncoder(handle_unknown='ignore', sparse=False),
            StackingClassifier(estimators=[
                ("rf", RandomForestClassifier(n_estimators=10,
                                              random_state=42)),
                ("gb", GradientBoostingClassifier(n_estimators=10,
                                                  random_state=42)),
                ("knn", KNeighborsClassifier(n_neighbors=2))
            ], final_estimator=LogisticRegression(), cv=2))

        X_train = pandas.DataFrame(
            dict(text=['A', 'B', 'A', 'B', 'AA', 'B',
                       'A', 'B', 'A', 'AA', 'B', 'B'],
                 val=[0.5, 0.6, 0.7, 0.61, 0.51, 0.67,
                      0.51, 0.61, 0.71, 0.611, 0.511, 0.671]))
        X_train['val'] = X_train.val.astype(numpy.float32)
        y_train = numpy.array([0, 1, 0, 1, 0, 1,
                               0, 1, 0, 1, 0, 1])
        pipeline.fit(X_train, y_train)
        with self.assertRaises(RuntimeError):
            to_onnx(pipeline, X=X_train[:1], target_opset=TARGET_OPSET)

    @unittest.skipIf(StackingClassifier is None,
                     reason="new in 0.22")
    def test_issue_786(self):
        pipeline = make_pipeline(
            OneHotEncoder(handle_unknown='ignore', sparse=False),
            StackingClassifier(estimators=[
                    ("rf", RandomForestClassifier(n_estimators=10,
                                                  random_state=42)),
                    ("gb", GradientBoostingClassifier(n_estimators=10,
                                                      random_state=42)),
                    ("knn", KNeighborsClassifier(n_neighbors=2))
                ], final_estimator=LogisticRegression(), cv=2))

        X_train = pandas.DataFrame(
            dict(text=['A', 'B', 'A', 'B', 'AA', 'B',
                       'A', 'B', 'A', 'AA', 'B', 'B'],
                 val=[0.5, 0.6, 0.7, 0.61, 0.51, 0.67,
                      0.51, 0.61, 0.71, 0.611, 0.511, 0.671]))
        X_train['val'] = (X_train.val * 1000).astype(numpy.float32)
        y_train = numpy.array([0, 1, 0, 1, 0, 1,
                               0, 1, 0, 1, 0, 1])
        pipeline.fit(X_train, y_train)
        onx = to_onnx(pipeline, X=X_train[:1],
                      options={'zipmap': False},
                      target_opset=TARGET_OPSET)
        # with open("ohe_debug.onnx", "wb") as f:
        #     f.write(onx.SerializeToString())
        sess = InferenceSession(onx.SerializeToString())
        res = sess.run(None, {'text': X_train.text.values.reshape((-1, 1)),
                              'val': X_train.val.values.reshape((-1, 1))})
        assert_almost_equal(pipeline.predict(X_train), res[0])
        assert_almost_equal(pipeline.predict_proba(X_train), res[1])


if __name__ == "__main__":
    # import logging
    # log = logging.getLogger('skl2onnx')
    # log.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestStackingConverter().test_issue_786()
    unittest.main()
