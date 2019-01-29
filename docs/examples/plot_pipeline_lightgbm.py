# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _example-lightgbm:

Convert a pipeline with a LightGbm model
========================================

*scikit-onnx* only converts *scikit-learn* models into *ONNX*
but many libraries implement *scikit-learn* API so that their models
can be included in a *scikit-learn* pipeline. This example considers
a pipeline including a *LightGbm* model. *scikit-onnx* can convert
the whole pipeline as long as it knows the converter associated to
a *LGBMClassifier*. Let's see how to do it.

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


##################################
# First try to convert
# ++++++++++++++++++++
#
# Obviously, it fails because the convert for *LGBMClassifier*
# is not registered.

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    model_onnx = convert_sklearn(pipe, 'pipeline',
                                 [('input', FloatTensorType([1, 2]))])
except Exception as e:
    print(e)


########################
# Register the converter for LGBMClassifier
# +++++++++++++++++++++++++++++++++++++++++
#
# The converter is implemented in *onnxmltools* and
# follows a different design than the current one
# of *scikit-onnx*. This will change in a short future.
# See also :ref:`l-register-converter`.
# First the converter:
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm # the 

###########################
# The shape calculator of onnxmltools must be adapted for our case.
# This will change in a short future.
import numbers
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType, SequenceType

def lightgbm_classifier_shape_extractor(operator):
    N = operator.inputs[0].type.shape[0]

    class_labels = operator.raw_operator.classes_
    if all(isinstance(i, numpy.ndarray) for i in class_labels):
        class_labels = numpy.concatenate(class_labels)
    if all(isinstance(i, str) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N])
        operator.outputs[1].type = SequenceType(DictionaryType(StringTensorType([]), FloatTensorType([])), N)
    elif all(isinstance(i, (numbers.Real, bool, numpy.bool_)) for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[N])
        operator.outputs[1].type = SequenceType(DictionaryType(Int64TensorType([]), FloatTensorType([])), N)
    else:
        raise ValueError('Unsupported or mixed label types')

###########################
# Let's register the new converter.
from skl2onnx import update_registered_converter
update_registered_converter(LGBMClassifier, 'LightGbmLGBMClassifier',                                    
                            lightgbm_classifier_shape_extractor,
                            convert_lightgbm)

##################################
# Convert again
# +++++++++++++

model_onnx = convert_sklearn(pipe, 'pipeline',
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
