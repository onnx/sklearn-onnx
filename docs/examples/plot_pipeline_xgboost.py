# SPDX-License-Identifier: Apache-2.0


"""
.. _example-xgboost:

Convert a pipeline with a XGBoost model
========================================

.. index:: XGBoost

*sklearn-onnx* only converts *scikit-learn* models into *ONNX*
but many libraries implement *scikit-learn* API so that their models
can be included in a *scikit-learn* pipeline. This example considers
a pipeline including a *XGBoost* model. *sklearn-onnx* can convert
the whole pipeline as long as it knows the converter associated to
a *XGBClassifier*. Let's see how to do it.

.. contents::
    :local:

Train a XGBoost classifier
++++++++++++++++++++++++++
"""
import os
import numpy
import matplotlib.pyplot as plt
import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnxruntime as rt
import sklearn
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost
from xgboost import XGBClassifier
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
import onnxmltools
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa
import onnxmltools.convert.common.data_types

data = load_iris()
X = data.data[:, :2]
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

pipe = Pipeline([('scaler', StandardScaler()),
                 ('lgbm', XGBClassifier(n_estimators=3))])
pipe.fit(X, y)

# The conversion fails but it is expected.

try:
    convert_sklearn(pipe, 'pipeline_xgboost',
                    [('input', FloatTensorType([None, 2]))],
                    target_opset=12)
except Exception as e:
    print(e)

# The error message tells no converter was found
# for XGBoost models. By default, *sklearn-onnx*
# only handles models from *scikit-learn* but it can
# be extended to every model following *scikit-learn*
# API as long as the module knows there exists a converter
# for every model used in a pipeline. That's why
# we need to register a converter.

######################################
# Register the converter for XGBClassifier
# ++++++++++++++++++++++++++++++++++++++++
#
# The converter is implemented in *onnxmltools*:
# `onnxmltools...XGBoost.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# xgboost/operator_converters/XGBoost.py>`_.
# and the shape calculator:
# `onnxmltools...Classifier.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# xgboost/shape_calculators/Classifier.py>`_.

##############################################
# Then we import the converter and shape calculator.

###########################
# Let's register the new converter.
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

##################################
# Display the ONNX graph
# ++++++++++++++++++++++

pydot_graph = GetPydotGraph(
    model_onnx.graph, name=model_onnx.graph.name, rankdir="TB",
    node_producer=GetOpNodeProducer(
        "docstring", color="yellow",
        fillcolor="yellow", style="filled"))
pydot_graph.write_dot("pipeline.dot")

os.system('dot -O -Gdpi=300 -Tpng pipeline.dot')

image = plt.imread("pipeline.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
print("onnxmltools: ", onnxmltools.__version__)
print("xgboost: ", xgboost.__version__)
