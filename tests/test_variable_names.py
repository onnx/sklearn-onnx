# SPDX-License-Identifier: Apache-2.0
# coding: utf-8

import unittest
import copy
from textwrap import dedent
from io import StringIO
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
except ImportError:
    InvalidArgument = RuntimeError
from skl2onnx import get_model_alias, update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from skl2onnx import convert_sklearn, to_onnx
from onnxconverter_common.data_types import (
    FloatTensorType, Int64TensorType, StringTensorType)
from test_utils import fit_regression_model, TARGET_OPSET


class Passthrough:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def parser(scope, model, inputs, custom_parsers=None):
    alias = get_model_alias(type(model))
    operator = scope.declare_local_operator(alias, model)
    operator.inputs = inputs
    for op_input in inputs:
        op_output = scope.declare_local_variable(
            op_input.raw_name, copy.deepcopy(op_input.type))
        operator.outputs.append(op_output)
    return operator.outputs


def shape_calculator(operator):
    op_input_map = {op_input.raw_name: op_input
                    for op_input in operator.inputs}
    for op_output in operator.outputs:
        op_output.type.shape = op_input_map[op_output.raw_name].type.shape


def converter(scope, operator, container):
    op_input_map = {op_input.raw_name: op_input
                    for op_input in operator.inputs}
    for op_output in operator.outputs:
        op_input = op_input_map[op_output.raw_name]
        OnnxIdentity(
            op_input,
            output_names=[op_output],
            op_version=container.target_opset,
        ).add_to(scope, container)


class TestVariableNames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        update_registered_converter(
            Passthrough, "Passthrough",
            shape_calculator, converter,
            parser=parser)

    def test_variable_names(self):
        pipeline = Pipeline([("passthrough", Passthrough())])
        initial_types = [("input", FloatTensorType([None, 2]))]
        model_onnx = convert_sklearn(pipeline, initial_types=initial_types,
                                     target_opset=TARGET_OPSET,
                                     verbose=0)
        self.assertIn('Identity', str(model_onnx))
        x = np.array([0, 1, 1, 0], dtype=np.float32).reshape((-1, 2))
        sess = InferenceSession(model_onnx.SerializeToString())
        name = sess.get_inputs()[0].name
        got = sess.run(None, {name: x})
        assert_almost_equal(x, got[0])

    def test_variable_names_distinct(self):
        pipeline = Pipeline([("passthrough", Passthrough())])
        initial_types = [("INPUTA", FloatTensorType([None, 2]))]
        final_types = [("OUTPUTA", FloatTensorType([None, 2]))]
        model_onnx = convert_sklearn(pipeline, initial_types=initial_types,
                                     target_opset=TARGET_OPSET,
                                     final_types=final_types,
                                     verbose=0)
        x = np.array([0, 1, 1, 0], dtype=np.float32).reshape((-1, 2))
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'INPUTA': x})
        assert_almost_equal(x, got[0])

    def test_variable_names_output(self):
        pipeline = Pipeline([("passthrough", Passthrough())])
        initial_types = [("input", FloatTensorType([None, 2]))]
        final_types = initial_types
        with self.assertRaises(RuntimeError):
            convert_sklearn(pipeline, initial_types=initial_types,
                            target_opset=TARGET_OPSET,
                            final_types=final_types)

    def _test_non_ascii_variable_name(self):
        model, X = fit_regression_model(LinearRegression())
        model_onnx = to_onnx(
            model, name="linear regression",
            initial_types=[("年齢", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        # Invalid Feed Input Name:\u5e74\u9f62
        # sess.run(None, {'年齢': X})
        self.assertTrue(sess is not None)

    def test_non_ascii_variable_name_pipeline(self):

        data = dedent("""
            pclass,survived,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,home.dest
            1,1,"A",female,29.0,0,0,24160,211.3375,B5,S,2,,"MO"
            1,1,"B",male,0.9167,1,2,113781,151.55,C22 C26,S,11,,"Can"
            1,0,"C",female,2.0,1,2,113781,151.55,C22 C26,S,,,"Can"
            1,0,"D",male,30.0,1,2,113781,151.55,C22 C26,S,,135.0,"Can"
            1,0,"E",female,25.0,1,2,113781,151.55,C22 C26,S,,,"Can"
            1,1,"F",male,48.0,0,0,19952,26.55,E12,S,3,,"NY"
            1,1,"G",female,63.0,1,0,13502,77.9583,D7,S,10,,"NY"
            1,0,"H",male,39.0,0,0,112050,0.0,A36,S,,,"NI"
            1,1,"I",female,53.0,2,0,11769,51.4792,C101,S,D,,"NY"
            1,0,"J",male,71.0,0,0,PC 17609,49.5042,,C,,22.0,"Uruguay"
            1,0,"K",male,47.0,1,0,PC 17757,227.525,C62 C64,C,,124.0,"NY"
            1,1,"L",female,18.0,1,0,PC 17757,227.525,C62 C64,C,4,,"NY"
            1,1,"M",female,24.0,0,0,PC 17477,69.3,B35,C,9,,"F"
            1,1,"N",female,26.0,0,0,19877,78.85,,S,6,,
            1,1,"L",male,80.0,0,0,27042,30.0,A23,S,B,,"Yorks"
            1,0,"O",male,,0,0,PC 17318,25.925,,S,,,"NY"
            1,0,"P",male,24.0,0,1,PC 17558,247.5208,B58 B60,C,,,"PQ"
            1,1,"Q",female,50.0,0,1,PC 17558,247.5208,B58 B60,C,6,,"PQ"
            1,1,"R",female,32.0,0,0,11813,76.2917,D15,C,8,,
            1,0,"S",male,36.0,0,0,13050,75.2417,C6,C,A,,"MN"
        """).strip(" \n")
        data = pd.read_csv(StringIO(data))
        data.rename(columns={"age": "年齢"}, inplace=True)
        X = data.drop('survived', axis=1)
        # y = data['survived']
        cols = ['embarked', 'sex', 'pclass', '年齢', 'fare']
        X = X[cols]
        for cat in ['embarked', 'sex', 'pclass']:
            X[cat].fillna('missing', inplace=True)
        numeric_features = ['年齢', 'fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        categorical_features = ['embarked', 'sex', 'pclass']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        preprocessor.fit_transform(X)
        initial_type = [('pclass', Int64TensorType(shape=[None, 1])),
                        ('sex', StringTensorType(shape=[None, 1])),
                        ('年齢', FloatTensorType(shape=[None, 1])),
                        ('fare', FloatTensorType(shape=[None, 1])),
                        ('embarked', StringTensorType(shape=[None, 1]))]

        onnx_object = convert_sklearn(
            preprocessor, initial_types=initial_type,
            target_opset=TARGET_OPSET)
        sess = InferenceSession(onnx_object.SerializeToString())
        self.assertTrue(sess is not None)
        # Invalid Feed Input Name:\u5e74\u9f62
        # onx_data = {}
        # for col in initial_type:
        #     onx_data[col[0]] = X[col[0]].values.reshape((-1, 1))
        # sess.run(None, onx_data)


if __name__ == "__main__":
    unittest.main()
