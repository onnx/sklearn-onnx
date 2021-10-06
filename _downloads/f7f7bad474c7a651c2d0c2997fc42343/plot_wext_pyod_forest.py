# SPDX-License-Identifier: Apache-2.0

"""
.. _example-pyod-iforest:

Converter for pyod.models.iforest.IForest
=========================================

.. index:: pyod, iforest

This example answers issues `685
<https://github.com/onnx/sklearn-onnx/issues/685>`_.
It implements a custom converter for model `pyod.models.iforest.IForest
<https://pyod.readthedocs.io/en/latest/
pyod.models.html#module-pyod.models.iforest>`_.
This example uses :ref:`l-plot-custom-converter` as a start.

.. contents::
    :local:

Trains a model
++++++++++++++

All imports. It also registered onnx converters for :epgk:`xgboost`
and *lightgbm*.
"""
import numpy as np
import pandas as pd
from onnxruntime import InferenceSession
from sklearn.preprocessing import MinMaxScaler
from skl2onnx.proto import onnx_proto
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, guess_numpy_type)
from skl2onnx import to_onnx, update_registered_converter, get_model_alias
from skl2onnx.algebra.onnx_ops import (
    OnnxIdentity, OnnxMul, OnnxLess, OnnxConcat, OnnxCast, OnnxAdd,
    OnnxClip)
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
try:
    from pyod.models.iforest import IForest
except (ValueError, ImportError) as e:
    print("Unable to import pyod:", e)
    IForest = None

if IForest is not None:
    data1 = {'First':  [500, 500, 400, 100, 200, 300, 100],
             'Second': ['a', 'b', 'a', 'b', 'a', 'b', 'c']}

    df1 = pd.DataFrame(data1, columns=['First', 'Second'])
    dumdf1 = pd.get_dummies(df1)
    scaler = MinMaxScaler()
    scaler.partial_fit(dumdf1)
    sc_data = scaler.transform(dumdf1)
    model1 = IForest(n_estimators=10, bootstrap=True, behaviour='new',
                     contamination=0.1, random_state=np.random.RandomState(42),
                     verbose=1, n_jobs=-1).fit(sc_data)
    feature_names2 = dumdf1.columns

    initial_type = [('float_input',
                     FloatTensorType([None, len(feature_names2)]))]


#############################################
# We check that the conversion fails as expected.

if IForest is not None:
    try:
        to_onnx(model1, initial_types=initial_type)
    except Exception as e:
        print(e)


####################################################
# Custom converter
# ++++++++++++++++
#
# First the parser and the shape calculator.
# The parser defines the number of outputs and their type.
# The shape calculator defines their dimensions.

def pyod_iforest_parser(scope, model, inputs, custom_parsers=None):
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)

    # inputs
    this_operator.inputs.append(inputs[0])

    # outputs
    cls_type = inputs[0].type.__class__
    val_y1 = scope.declare_local_variable('label', Int64TensorType())
    val_y2 = scope.declare_local_variable('probability', cls_type())
    this_operator.outputs.append(val_y1)
    this_operator.outputs.append(val_y2)

    # end
    return this_operator.outputs


def pyod_iforest_shape_calculator(operator):
    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type.shape = [N, 1]
    operator.outputs[1].type.shape = [N, 2]

############################################
# Then the converter.


def pyod_iforest_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    # In most case, computation happen in floats.
    # But it might be with double. ONNX is very strict
    # about types, every constant should have the same
    # type as the input.
    dtype = guess_numpy_type(X.type)

    detector = op.detector_  # Should be IForest from scikit-learn.
    lab_pred = OnnxSubEstimator(detector, X, op_version=opv)
    scores = OnnxIdentity(lab_pred[1], op_version=opv)

    # labels
    threshold = op.threshold_
    above = OnnxLess(scores, np.array([threshold], dtype=dtype),
                     op_version=opv)
    labels = OnnxCast(above, op_version=opv, to=onnx_proto.TensorProto.INT64,
                      output_names=out[:1])

    # probabilities
    train_scores = op.decision_scores_
    scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
    scores_ = OnnxMul(scores, np.array([-1], dtype=dtype),
                      op_version=opv)
    print(scaler.min_)
    print(scaler.scale_)

    scaled = OnnxMul(scores_, scaler.scale_.astype(dtype), op_version=opv)
    scaled_centered = OnnxAdd(scaled, scaler.min_.astype(dtype),
                              op_version=opv)
    clipped = OnnxClip(scaled_centered, np.array([0], dtype=dtype),
                       np.array([1], dtype=dtype),
                       op_version=opv)
    clipped_ = OnnxAdd(
        OnnxMul(clipped, np.array([-1], dtype=dtype),
                op_version=opv),
        np.array([1], dtype=dtype),
        op_version=opv)

    scores_2d = OnnxConcat(clipped_, clipped, axis=1, op_version=opv,
                           output_names=out[1:])

    labels.add_to(scope, container)
    scores_2d.add_to(scope, container)

########################################
# Finally the registration.


if IForest is not None:
    update_registered_converter(
        IForest, "PyodIForest",
        pyod_iforest_shape_calculator,
        pyod_iforest_converter,
        parser=pyod_iforest_parser)

#############################################
# And the conversion.

if IForest is not None:
    onx = to_onnx(model1, initial_types=initial_type,
                  target_opset=14)

###############################################
# Checking discrepencies
# ++++++++++++++++++++++

if IForest is not None:
    data = sc_data.astype(np.float32)

    expected_labels = model1.predict(data)
    expected_proba = model1.predict_proba(data)

    sess = InferenceSession(onx.SerializeToString())
    res = sess.run(None, {'float_input': data})

    onx_labels = res[0]
    onx_proba = res[1]

    diff_labels = np.abs(onx_labels.ravel() - expected_labels.ravel()).max()
    diff_proba = np.abs(onx_proba.ravel() - expected_proba.ravel()).max()

    print("dicrepencies:", diff_labels, diff_proba)

    print("ONNX labels", onx_labels)
    print("ONNX probabilities", onx_proba)
