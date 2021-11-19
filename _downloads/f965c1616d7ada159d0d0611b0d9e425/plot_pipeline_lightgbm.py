# SPDX-License-Identifier: Apache-2.0


"""
.. _example-lightgbm-pipe:

Convert a pipeline with a LightGbm model
========================================

.. index:: LightGbm

*sklearn-onnx* only converts *scikit-learn* models into *ONNX*
but many libraries implement *scikit-learn* API so that their models
can be included in a *scikit-learn* pipeline. This example considers
a pipeline including a *LightGbm* model. *sklearn-onnx* can convert
the whole pipeline as long as it knows the converter associated to
a *LGBMClassifier*. Let's see how to do it.

.. contents::
    :local:

Train a LightGBM classifier
+++++++++++++++++++++++++++
"""
import lightgbm
import onnxmltools
import skl2onnx
import onnx
import sklearn
import matplotlib.pyplot as plt
import os
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnxruntime as rt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm  # noqa
import onnxmltools.convert.common.data_types
from skl2onnx.common.data_types import FloatTensorType
import numpy
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

data = load_iris()
X = data.data[:, :2]
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

pipe = Pipeline([('scaler', StandardScaler()),
                 ('lgbm', LGBMClassifier(n_estimators=3))])
pipe.fit(X, y)

######################################
# Register the converter for LGBMClassifier
# +++++++++++++++++++++++++++++++++++++++++
#
# The converter is implemented in *onnxmltools*:
# `onnxmltools...LightGbm.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# lightgbm/operator_converters/LightGbm.py>`_.
# and the shape calculator:
# `onnxmltools...Classifier.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# lightgbm/shape_calculators/Classifier.py>`_.

##############################################
# Then we import the converter and shape calculator.

###########################
# Let's register the new converter.
update_registered_converter(
    LGBMClassifier, 'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes, convert_lightgbm,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

##################################
# Convert again
# +++++++++++++

model_onnx = convert_sklearn(
    pipe, 'pipeline_lightgbm',
    [('input', FloatTensorType([None, 2]))],
    target_opset=12)

# And save.
with open("pipeline_lightgbm.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

###########################
# Compare the predictions
# +++++++++++++++++++++++
#
# Predictions with LightGbm.

print("predict", pipe.predict(X[:5]))
print("predict_proba", pipe.predict_proba(X[:1]))

##########################
# Predictions with onnxruntime.

try:
    sess = rt.InferenceSession("pipeline_lightgbm.onnx")
except OrtFail as e:
    print(e)
    print("The converter requires onnxmltools>=1.7.0")
    sess = None

if sess is not None:
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
print("lightgbm: ", lightgbm.__version__)
