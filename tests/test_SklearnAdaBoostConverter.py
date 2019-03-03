# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnAdaBoostModels(unittest.TestCase):

    def test_ada_boost_classifier_samme_r(self):
        data = load_digits()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = AdaBoostClassifier(n_estimators=10, algorithm='SAMME.R')
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'AdaBoost classification',
                                     [('input',
                                      FloatTensorType(X_test.shape))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X_test.astype('float32'), model, model_onnx,
                            basename="SklearnAdaBoostClassifierSAMMER")

    def test_ada_boost_classifier_samme(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = AdaBoostClassifier(n_estimators=15, algorithm='SAMME')
        model.fit(X_train, y_train)
        model_onnx = convert_sklearn(model, 'AdaBoost classification',
                                     [('input',
                                      FloatTensorType(X_train.shape))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X_test.astype('float32'), model, model_onnx,
                            basename="SklearnAdaBoostClassifierSAMME")


if __name__ == "__main__":
    unittest.main()
