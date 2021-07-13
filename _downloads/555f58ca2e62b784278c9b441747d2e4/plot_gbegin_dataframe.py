# SPDX-License-Identifier: Apache-2.0

"""
Dataframe as an input
=====================

.. index:: dataframe

A pipeline usually ingests data as a matrix. It may be converted in a matrix
if all the data share the same type. But data held in a dataframe
have usually multiple types, float, integer or string for categories.
ONNX also supports that case.

.. contents::
    :local:

A dataset with categories
+++++++++++++++++++++++++

"""
from mlinsights.plotting import pipeline2dot
import numpy
import pprint
from mlprodict.onnx_conv import guess_schema_from_data
from onnxruntime import InferenceSession
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import to_onnx as to_onnx_ext
from skl2onnx import to_onnx
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


data = DataFrame([
    dict(CAT1='a', CAT2='c', num1=0.5, num2=0.6, y=0),
    dict(CAT1='b', CAT2='d', num1=0.4, num2=0.8, y=1),
    dict(CAT1='a', CAT2='d', num1=0.5, num2=0.56, y=0),
    dict(CAT1='a', CAT2='d', num1=0.55, num2=0.56, y=1),
    dict(CAT1='a', CAT2='c', num1=0.35, num2=0.86, y=0),
    dict(CAT1='a', CAT2='c', num1=0.5, num2=0.68, y=1),
])

cat_cols = ['CAT1', 'CAT2']
train_data = data.drop('y', axis=1)


categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_cols)],
    remainder='passthrough')
pipe = Pipeline([('preprocess', preprocessor),
                 ('rf', RandomForestClassifier())])
pipe.fit(train_data, data['y'])

#####################################
# Display.

dot = pipeline2dot(pipe, train_data)
ax = plot_graphviz(dot)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

#######################################
# Conversion to ONNX
# ++++++++++++++++++
#
# Function *to_onnx* does not handle dataframes.


try:
    onx = to_onnx(pipe, train_data[:1])
except NotImplementedError as e:
    print(e)

###################################
# But it possible to use an extended one.


onx = to_onnx_ext(
    pipe, train_data[:1],
    options={RandomForestClassifier: {'zipmap': False}})

#######################################
# Graph
# +++++


oinf = OnnxInference(onx)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


#################################
# Prediction with ONNX
# ++++++++++++++++++++
#
# *onnxruntime* does not support dataframes.


sess = InferenceSession(onx.SerializeToString())
try:
    sess.run(None, train_data)
except Exception as e:
    print(e)

###########################
# Let's use a shortcut

oinf = OnnxInference(onx)
got = oinf.run(train_data)
print(pipe.predict(train_data))
print(got['label'])

#################################
# And probilities.

print(pipe.predict_proba(train_data))
print(got['probabilities'])

######################################
# It looks ok. Let's dig into the details to
# directly use *onnxruntime*.
#
# Unhide conversion logic with a dataframe
# ++++++++++++++++++++++++++++++++++++++++
#
# A dataframe can be seen as a set of columns with
# different types. That's what ONNX should see:
# a list of inputs, the input name is the column name,
# the input type is the column type.


init = guess_schema_from_data(train_data)

pprint.pprint(init)

###############################
# Let's use float instead.


for c in train_data.columns:
    if c not in cat_cols:
        train_data[c] = train_data[c].astype(numpy.float32)

init = guess_schema_from_data(train_data)
pprint.pprint(init)

##############################
# Let's convert with *skl2onnx* only.

onx2 = to_onnx(
    pipe, initial_types=init,
    options={RandomForestClassifier: {'zipmap': False}})

#####################################
# Let's run it with onnxruntime.
# We need to convert the dataframe into a dictionary
# where column names become keys, and column values become
# values.

inputs = {c: train_data[c].values.reshape((-1, 1))
          for c in train_data.columns}
pprint.pprint(inputs)

#############################
# Inference.

sess2 = InferenceSession(onx2.SerializeToString())

got2 = sess2.run(None, inputs)

print(pipe.predict(train_data))
print(got2[0])

#################################
# And probilities.

print(pipe.predict_proba(train_data))
print(got2[1])
