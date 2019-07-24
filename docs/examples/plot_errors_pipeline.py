# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _errors-pipeline:

Errors while converting a pipeline
==================================

A pipeline is a patchwork of many different pieces
and the probability of the first try to convert it fails
is quite high. This script gathers the most frequent one
and suggest a solution.


.. contents::
    :local:

Converter not registered
++++++++++++++++++++++++

*LightGBM* implements random forest which follow
*scikit-learn* API. Due to that, they can be included a
*scikit-learn* pipeline which can be used to optimize
hyperparameters in grid search or to validate the model
with a cross validation. However, *sklearn-onnx* does not
implement a converter for an instance of
`LGBMClassifier
<https://lightgbm.readthedocs.io/en/latest/Python-API.html?
highlight=LGBMClassifier#lightgbm.LGBMClassifier>`_.
Let's see what happens when a simple pipeline is being converted.
"""
import skl2onnx
import onnxruntime
import onnx
import sklearn
from pandas import DataFrame
import matplotlib.pyplot as plt
import os
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
from skl2onnx.common.data_types import StringTensorType
from sklearn.linear_model import LogisticRegression
from skl2onnx import update_registered_converter
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm  # noqa
from skl2onnx.common.data_types import DictionaryType, SequenceType
import numbers
from skl2onnx import convert_sklearn
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
                 ('lgbm', LGBMClassifier(n_estimators=1, max_depth=1))])
pipe.fit(X, y)

##################################
# The conversion happens here and fails.


try:
    model_onnx = convert_sklearn(
        pipe, 'pipeline', [('input', FloatTensorType([None, 2]))])
except Exception as e:
    print(e)

###################################
# *sklearn-onnx* needs to know the appropriate converter
# for class *LGBMClassifier*, the converter needs to be registered.
# The converter comes with two pieces: a shape calculator which
# computes output shapes based on inputs shapes and the converter
# itself which extracts the coefficients of the random forest
# and converts them into *ONNX* format.
# First, the shape calculator:


def lightgbm_classifier_shape_extractor(operator):
    N = operator.inputs[0].type.shape[0]

    class_labels = operator.raw_operator.classes_
    if all(isinstance(i, numpy.ndarray) for i in class_labels):
        class_labels = numpy.concatenate(class_labels)
    if all(isinstance(i, str) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N])
        operator.outputs[1].type = SequenceType(
            DictionaryType(StringTensorType([]), FloatTensorType([])), N)
    elif all(isinstance(i, (numbers.Real, bool,
                            numpy.bool_)) for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[N])
        operator.outputs[1].type = SequenceType(
            DictionaryType(Int64TensorType([]), FloatTensorType([])), N)
    else:
        raise ValueError('Unsupported or mixed label types.')


###################################
# Then the converter itself:


###################################
# They are both registered with the following instruction.
update_registered_converter(LGBMClassifier, 'LightGbmLGBMClassifier',
                            lightgbm_classifier_shape_extractor,
                            convert_lightgbm)


#################################
# Let's convert again.

model_onnx = convert_sklearn(
    pipe, 'pipeline', [('input', FloatTensorType([None, 2]))])

print(str(model_onnx)[:300] + "\n...")

##################################
# .. _l-dataframe-initial-type:
#
# Working with dataframes
# +++++++++++++++++++++++
#
# *sklearn-onnx* converts a pipeline without knowing the training data,
# more specifically, it does not know the input variables. This is why
# it complain when the parameter *initial_type* is not filled
# when function :func:`skl2onnx.convert_sklearn`
# is called. Let's see what happens without it.


data = load_iris()
X = data.data[:, :2]
y = data.target

clf = LogisticRegression()
clf.fit(X, y)

try:
    model_onnx = convert_sklearn(clf)
except Exception as e:
    print(e)

################################
# We need to define the initial type.
# Let's write some code to automatically
# fill that parameter from a dataframe.


def convert_dataframe_schema(df, drop=None):
    inputs = []
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([None, 1])
        elif v == 'float64':
            t = FloatTensorType([None, 1])
        else:
            t = StringTensorType([None, 1])
        inputs.append((k, t))
    return inputs


data = DataFrame(X, columns=["X1", "X2"])

inputs = convert_dataframe_schema(data)
print(inputs)

##################################
# Let's convert again.

try:
    model_onnx = convert_sklearn(clf, initial_types=inputs)
except Exception as e:
    print(e)

##################################
# *sklean-onnx* tells it cannot match two single inputs
# with one input vector of dimension 2.
# Let's try it that way:

model_onnx = convert_sklearn(
    clf, initial_types=[('X', FloatTensorType([None, 2]))])
print(str(model_onnx)[:300] + "\n...")

##################################
# What if now this model is included in a pipeline
# with a `ColumnTransformer
# <https://scikit-learn.org/stable/modules/generated/
# sklearn.compose.ColumnTransformer.html>`_.
# The following pipeline is a way to concatenate multiple
# columns into a single one with a
# `FunctionTransformer
# <https://scikit-learn.org/stable/modules/generated/
# sklearn.preprocessing.FunctionTransformer.html>`_
# with identify function.

pipe = Pipeline(steps=[
    ('select', ColumnTransformer(
        [('id', FunctionTransformer(), ['X1', 'X2'])])),
    ('logreg', clf)
])
pipe.fit(data[['X1', 'X2']], y)

pipe_onnx = convert_sklearn(pipe, initial_types=inputs)
print(str(pipe_onnx)[:300] + "\n...")

#########################################
# Let's draw the pipeline for a better understanding.


pydot_graph = GetPydotGraph(
    pipe_onnx.graph, name=model_onnx.graph.name, rankdir="TB",
    node_producer=GetOpNodeProducer(
        "docstring", color="orange", fillcolor="orange", style="filled"))
pydot_graph.write_dot("pipeline_concat.dot")

os.system('dot -O -Gdpi=300 -Tpng pipeline_concat.dot')

image = plt.imread("pipeline_concat.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')


##################################
# Unused inputs
# +++++++++++++
#
# *sklearn-onnx* converts a model into a ONNX graph
# and this graph is then used to compute predictions
# with a backend. The smaller the graph is, the faster
# the computation is. That's why *sklearn-onnx* raises some
# exception when it detects when something can be optimized.
# That's the case when more inputs than needed are declared.
# Let's reuse the previous example with a new dummy feature.

data["dummy"] = 4.5
inputs = convert_dataframe_schema(data)
print(inputs)

####################################
# The new *initial_types* makes the conversion fail.

try:
    pipe_onnx = convert_sklearn(pipe, initial_types=inputs)
except Exception as e:
    print(e)

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
