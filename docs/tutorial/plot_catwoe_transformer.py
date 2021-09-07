# SPDX-License-Identifier: Apache-2.0

"""
.. _example-catwoe-transformer:

Converter for WOEEncoder from categorical_encoder
=================================================

`WOEEncoder <https://contrib.scikit-learn.org/category_encoders/woe.html>`_
is a transformer implemented in `categorical_encoder
<https://contrib.scikit-learn.org/category_encoders/>`_ and as such,
any converter would not be included in *sklearn-onnx* which only
implements converters for *scikit-learn* models. Anyhow, this
example demonstrates how to implement a custom converter
for *WOEEncoder*. This code is not fully tested for all possible
cases the original encoder can handle.

.. index:: WOE, WOEEncoder

.. contents::
    :local:

A simple example
++++++++++++++++

Let's take the `Iris dataset
<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html>`_.
Every feature is converter into integer.
"""
import numpy as np
from onnxruntime import InferenceSession
from sklearn.datasets import load_iris
from sklearn.preprocessing import OrdinalEncoder as SklOrdinalEncoder
from category_encoders import WOEEncoder, OrdinalEncoder
from skl2onnx import update_registered_converter, to_onnx, get_model_alias
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.utils import check_input_and_output_numbers
from skl2onnx.algebra.onnx_ops import OnnxCast
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx.sklapi import WOETransformer
import skl2onnx.sklapi.register  # noqa

data = load_iris()
X, y = data.data, data.target
X = X.astype(np.int64)[:, :2]
y = (y == 2).astype(np.int64)

woe = WOEEncoder(cols=[0]).fit(X, y)
print(woe.transform(X[:5]))

########################################
# Let's look into the trained parameters of the model.
# It appears that WOEEncoder uses an OrdinalEncoder
# but not the one from scikit-learn. We need to add a
# converter for this model tool.

print("encoder", type(woe.ordinal_encoder), woe.ordinal_encoder)
print("mapping", woe.mapping)
print("encoder.mapping", woe.ordinal_encoder.mapping)
print("encoder.cols", woe.ordinal_encoder.cols)

######################################
# Custom converter for OrdinalEncoder
# +++++++++++++++++++++++++++++++++++
#
# We start from example :ref:`l-plot-custom-converter`
# and then write the conversion.


def ordenc_to_sklearn(op_mapping):
    "Converts OrdinalEncoder mapping to scikit-learn OrdinalEncoder."
    cats = []
    for column_map in op_mapping:
        col = column_map['col']
        while len(cats) <= col:
            cats.append(None)
        mapping = column_map['mapping']
        res = []
        for i in range(mapping.shape[0]):
            if np.isnan(mapping.index[i]):
                continue
            ind = mapping.iloc[i]
            while len(res) <= ind:
                res.append(0)
            res[ind] = mapping.index[i]
        cats[col] = np.array(res, dtype=np.int64)

    skl_ord = SklOrdinalEncoder(categories=cats, dtype=np.int64)
    skl_ord.categories_ = cats
    return skl_ord


def ordinal_encoder_shape_calculator(operator):
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=1)
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].get_first_dimension()
    shape = operator.inputs[0].type.shape
    second_dim = None if len(shape) != 2 else shape[1]
    output_type = input_type([input_dim, second_dim])
    operator.outputs[0].type = output_type


def ordinal_encoder_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    X = operator.inputs[0]

    skl_ord = ordenc_to_sklearn(op.mapping)
    cat = OnnxSubEstimator(skl_ord, X, op_version=opv,
                           output_names=operator.outputs[:1])
    cat.add_to(scope, container)


update_registered_converter(
    OrdinalEncoder, "CategoricalEncoderOrdinalEncoder",
    ordinal_encoder_shape_calculator,
    ordinal_encoder_converter)


###################################
# Let's compute the output one a short example.


enc = OrdinalEncoder(cols=[0, 1])
enc.fit(X)
print(enc.transform(X[:5]))


###################################
# Let's check the ONNX conversion produces the same results.


ord_onx = to_onnx(enc, X[:1], target_opset=14)
sess = InferenceSession(ord_onx.SerializeToString())
print(sess.run(None, {'X': X[:5]})[0])

######################################
# That works.
#
# Custom converter for WOEEncoder
# +++++++++++++++++++++++++++++++
#
# We start from example :ref:`l-plot-custom-converter`
# and then write the conversion.


def woeenc_to_sklearn(op_mapping):
    "Converts WOEEncoder mapping to scikit-learn OrdinalEncoder."
    cats = []
    ws = []
    for column_map in op_mapping.items():
        col = column_map[0]
        while len(cats) <= col:
            cats.append('passthrough')
            ws.append(None)
        mapping = column_map[1]
        intervals = []
        weights = []
        for i in range(mapping.shape[0]):
            ind = mapping.index[i]
            if ind < 0:
                continue
            intervals.append((float(ind - 1), float(ind), False, True))
            weights.append(mapping.iloc[i])
        cats[col] = intervals
        ws[col] = weights

    skl = WOETransformer(intervals=cats, weights=ws, onehot=False)
    skl.fit(None)
    return skl


def woe_encoder_parser(
        scope, model, inputs, custom_parsers=None):
    if len(inputs) != 1:
        raise RuntimeError(
            "Unexpected number of inputs: %d != 1." % len(inputs))
    if inputs[0].type is None:
        raise RuntimeError(
            "Unexpected type: %r." % (inputs[0], ))
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs.append(inputs[0])
    this_operator.outputs.append(
        scope.declare_local_variable('catwoe', FloatTensorType()))
    return this_operator.outputs


def woe_encoder_shape_calculator(operator):
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=1)
    input_dim = operator.inputs[0].get_first_dimension()
    shape = operator.inputs[0].type.shape
    second_dim = None if len(shape) != 2 else shape[1]
    output_type = FloatTensorType([input_dim, second_dim])
    operator.outputs[0].type = output_type


def woe_encoder_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    X = operator.inputs[0]

    sub = OnnxSubEstimator(op.ordinal_encoder, X,
                           op_version=opv)
    cast = OnnxCast(sub, op_version=opv, to=np.float32)
    skl_ord = woeenc_to_sklearn(op.mapping)
    cat = OnnxSubEstimator(skl_ord, cast, op_version=opv,
                           output_names=operator.outputs[:1],
                           input_types=[FloatTensorType()])
    cat.add_to(scope, container)


update_registered_converter(
    WOEEncoder, "CategoricalEncoderWOEEncoder",
    woe_encoder_shape_calculator,
    woe_encoder_converter,
    parser=woe_encoder_parser)


###################################
# Let's compute the output one a short example.

woe = WOEEncoder(cols=[0, 1]).fit(X, y)
print(woe.transform(X[:5]))


###################################
# Let's check the ONNX conversion produces the same results.


woe_onx = to_onnx(woe, X[:1], target_opset=14)
sess = InferenceSession(woe_onx.SerializeToString())
print(sess.run(None, {'X': X[:5]})[0])
