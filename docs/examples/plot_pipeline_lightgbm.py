# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _example-lightgbm:

Convert a pipeline with a LightGbm model
========================================

.. index:: LightGbm

*sklearn-onnx* only converts *scikit-learn* models into *ONNX*
but many libraries implement *scikit-learn* API so that their models
can be included in a *scikit-learn* pipeline. This example considers
a pipeline including a *LightGbm* model. *sklearn-onnx* can convert
the whole pipeline as long as it knows the converter associated to
a *LGBMClassifier*. Let's see how to do it.

A couple of errors might happen while trying to convert
your own pipeline, some of them are described
and explained in :ref:`errors-pipeline`.

.. contents::
    :local:

Train a LightGBM classifier
+++++++++++++++++++++++++++
"""
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
# The current implementation has duplicated code which we replace
# by the implementation from *skl2onnx*.
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType, SequenceType
import onnxmltools.convert.common.data_types
onnxmltools.convert.common.data_types.Int64TensorType = Int64TensorType
onnxmltools.convert.common.data_types.StringTensorType = StringTensorType
onnxmltools.convert.common.data_types.FloatTensorType = FloatTensorType
onnxmltools.convert.common.data_types.DictionaryType = DictionaryType
onnxmltools.convert.common.data_types.SequenceType = SequenceType

##############################################
# Then we import the converter and shape calculator.
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from onnxmltools.convert.lightgbm.shape_calculators.Classifier import calculate_linear_classifier_output_shapes

###########################
# Let's register the new converter.
from skl2onnx import update_registered_converter
update_registered_converter(LGBMClassifier, 'LightGbmLGBMClassifier',                                    
                            calculate_linear_classifier_output_shapes,
                            convert_lightgbm)

##################################
# Convert again
# +++++++++++++

from skl2onnx import convert_sklearn
model_onnx = convert_sklearn(pipe, 'pipeline_lightgbm',
                             [('input', FloatTensorType([1, 2]))])

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

import onnxruntime as rt
import numpy
sess = rt.InferenceSession("pipeline_lightgbm.onnx")
pred_onx = sess.run(None, {"input": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])

##################################
# Display the ONNX graph
# ++++++++++++++++++++++

from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
pydot_graph = GetPydotGraph(model_onnx.graph, name=model_onnx.graph.name, rankdir="TB",
                            node_producer=GetOpNodeProducer("docstring", color="yellow",
                                                            fillcolor="yellow", style="filled"))
pydot_graph.write_dot("pipeline.dot")

import os
os.system('dot -O -Gdpi=300 -Tpng pipeline.dot')

import matplotlib.pyplot as plt
image = plt.imread("pipeline.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')

#################################
# **Versions used for this example**

import numpy, sklearn
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
import onnx, onnxruntime, skl2onnx, onnxmltools, lightgbm
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
print("onnxmltools: ", onnxmltools.__version__)
print("lightgbm: ", lightgbm.__version__)
