# SPDX-License-Identifier: Apache-2.0


"""
.. _example-xgboost:

Convert a pipeline with a XGBoost model
========================================

.. index:: XGBoost

:epkg:`sklearn-onnx` only converts :epkg:`scikit-learn` models
into :epkg:`ONNX` but many libraries implement :epkg:`scikit-learn`
API so that their models can be included in a :epkg:`scikit-learn`
pipeline. This example considers a pipeline including a :epkg:`XGBoost`
model. :epkg:`sklearn-onnx` can convert the whole pipeline as long as
it knows the converter associated to a *XGBClassifier*. Let's see
how to do it.

.. contents::
    :local:

Train a XGBoost classifier
++++++++++++++++++++++++++
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
import numpy
import onnxruntime as rt
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor, DMatrix, train as train_xgb
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster


data = load_iris()
X = data.data[:, :2]
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

pipe = Pipeline([('scaler', StandardScaler()),
                 ('xgb', XGBClassifier(n_estimators=3))])
pipe.fit(X, y)

# The conversion fails but it is expected.

try:
    convert_sklearn(pipe, 'pipeline_xgboost',
                    [('input', FloatTensorType([None, 2]))],
                    target_opset=12)
except Exception as e:
    print(e)

# The error message tells no converter was found
# for :epkg:`XGBoost` models. By default, :epkg:`sklearn-onnx`
# only handles models from :epkg:`scikit-learn` but it can
# be extended to every model following :epkg:`scikit-learn`
# API as long as the module knows there exists a converter
# for every model used in a pipeline. That's why
# we need to register a converter.

######################################
# Register the converter for XGBClassifier
# ++++++++++++++++++++++++++++++++++++++++
#
# The converter is implemented in :epkg:`onnxmltools`:
# `onnxmltools...XGBoost.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# xgboost/operator_converters/XGBoost.py>`_.
# and the shape calculator:
# `onnxmltools...Classifier.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# xgboost/shape_calculators/Classifier.py>`_.

update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

##################################
# Convert again
# +++++++++++++

model_onnx = convert_sklearn(
    pipe, 'pipeline_xgboost',
    [('input', FloatTensorType([None, 2]))],
    target_opset=12)

# And save.
with open("pipeline_xgboost.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

###########################
# Compare the predictions
# +++++++++++++++++++++++
#
# Predictions with XGBoost.

print("predict", pipe.predict(X[:5]))
print("predict_proba", pipe.predict_proba(X[:1]))

##########################
# Predictions with onnxruntime.

sess = rt.InferenceSession("pipeline_xgboost.onnx")
pred_onx = sess.run(None, {"input": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])

#############################
# Final graph
# +++++++++++


oinf = OnnxInference(model_onnx)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


#######################################
# Same example with XGBRegressor
# ++++++++++++++++++++++++++++++

update_registered_converter(
    XGBRegressor, 'XGBoostXGBRegressor',
    calculate_linear_regressor_output_shapes, convert_xgboost)


data = load_diabetes()
x = data.data
y = data.target
X_train, X_test, y_train, _ = train_test_split(x, y, test_size=0.5)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('xgb', XGBRegressor(n_estimators=3))])
pipe.fit(X_train, y_train)

print("predict", pipe.predict(X_test[:5]))

#############################
# ONNX

onx = to_onnx(pipe, X_train.astype(numpy.float32))

sess = rt.InferenceSession(onx.SerializeToString())
pred_onx = sess.run(None, {"X": X_test[:5].astype(numpy.float32)})
print("predict", pred_onx[0].ravel())

#################################
# Some discrepencies may appear. In that case,
# you should read :ref:`l-example-discrepencies-float-double`.

#################################################
# Same with a Booster
# +++++++++++++++++++
#
# A booster cannot be inserted in a pipeline. It requires
# a different conversion function because it does not
# follow :epkg:`scikit-learn` API.

x, y = make_classification(n_classes=2, n_features=5,
                           n_samples=100,
                           random_state=42, n_informative=3)
X_train, X_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                               random_state=42)

dtrain = DMatrix(X_train, label=y_train)

param = {'objective': 'multi:softmax', 'num_class': 3}
bst = train_xgb(param, dtrain, 10)

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onx = convert_xgboost_booster(bst, "name", initial_types=initial_type)

sess = rt.InferenceSession(onx.SerializeToString())
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
