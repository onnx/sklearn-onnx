# SPDX-License-Identifier: Apache-2.0


"""
Walk through intermediate outputs
=================================

We reuse the example :ref:`example-complex-pipeline` and
walk through intermediates outputs. It is very likely a converted
model gives different outputs or fails due to a custom
converter which is not correctly implemented.
One option is to look into the output of every node of the
ONNX graph.

.. contents::
    :local:

Create and train a complex pipeline
+++++++++++++++++++++++++++++++++++

We reuse the pipeline implemented in example
`Column Transformer with Mixed Types
<https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py>`_.
There is one change because
`ONNX-ML Imputer
<https://github.com/onnx/onnx/blob/master/docs/
Operators-ml.md#ai.onnx.ml.Imputer>`_
does not handle string type. This cannot be part of the final ONNX pipeline
and must be removed. Look for comment starting with ``---`` below.
"""
import skl2onnx
import onnx
import sklearn
import matplotlib.pyplot as plt
import os
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import numpy
import onnxruntime as rt
from skl2onnx import convert_sklearn
import pprint
from skl2onnx.common.data_types import (
    FloatTensorType, StringTensorType, Int64TensorType)
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)
X = data.drop('survived', axis=1)
y = data['survived']

# SimpleImputer on string is not available
# for string in ONNX-ML specifications.
# So we do it beforehand.
for cat in ['embarked', 'sex', 'pclass']:
    X[cat].fillna('missing', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    # --- SimpleImputer is not available for strings in ONNX-ML specifications.
    # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

clf.fit(X_train, y_train)

##################################
# Define the inputs of the ONNX graph
# +++++++++++++++++++++++++++++++++++
#
# *sklearn-onnx* does not know the features used to train the model
# but it needs to know which feature has which name.
# We simply reuse the dataframe column definition.
print(X_train.dtypes)

#########################
# After conversion.


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


inputs = convert_dataframe_schema(X_train)

pprint.pprint(inputs)

#############################
# Merging single column into vectors is not
# the most efficient way to compute the prediction.
# It could be done before converting the pipeline into a graph.

##################################
# Convert the pipeline into ONNX
# ++++++++++++++++++++++++++++++

try:
    model_onnx = convert_sklearn(clf, 'pipeline_titanic', inputs,
                                 target_opset=12)
except Exception as e:
    print(e)

#################################
# *scikit-learn* does implicit conversions when it can.
# *sklearn-onnx* does not. The ONNX version of *OneHotEncoder*
# must be applied on columns of the same type.

X_train['pclass'] = X_train['pclass'].astype(str)
X_test['pclass'] = X_test['pclass'].astype(str)
white_list = numeric_features + categorical_features
to_drop = [c for c in X_train.columns if c not in white_list]
inputs = convert_dataframe_schema(X_train, to_drop)

model_onnx = convert_sklearn(clf, 'pipeline_titanic', inputs,
                             target_opset=12)


# And save.
with open("pipeline_titanic.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

###########################
# Compare the predictions
# +++++++++++++++++++++++
#
# Final step, we need to ensure the converted model
# produces the same predictions, labels and probabilities.
# Let's start with *scikit-learn*.

print("predict", clf.predict(X_test[:5]))
print("predict_proba", clf.predict_proba(X_test[:1]))

##########################
# Predictions with onnxruntime.
# We need to remove the dropped columns and to change
# the double vectors into float vectors as *onnxruntime*
# does not support double floats.
# *onnxruntime* does not accept *dataframe*.
# inputs must be given as a list of dictionary.
# Last detail, every column was described  not really as a vector
# but as a matrix of one column which explains the last line
# with the *reshape*.

X_test2 = X_test.drop(to_drop, axis=1)
inputs = {c: X_test2[c].values for c in X_test2.columns}
for c in numeric_features:
    inputs[c] = inputs[c].astype(np.float32)
for k in inputs:
    inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))

################################
# We are ready to run *onnxruntime*.

sess = rt.InferenceSession("pipeline_titanic.onnx")
pred_onx = sess.run(None, inputs)
print("predict", pred_onx[0][:5])
print("predict_proba", pred_onx[1][:1])


####################################
# Compute intermediate outputs
# ++++++++++++++++++++++++++++
#
# Unfortunately, there is actually no way to ask
# *onnxruntime* to retrieve the output of intermediate nodes.
# We need to modifies the *ONNX* before it is given to *onnxruntime*.
# Let's see first the list of intermediate output.

model_onnx = load_onnx_model("pipeline_titanic.onnx")
for out in enumerate_model_node_outputs(model_onnx):
    print(out)

################################
# Not that easy to tell which one is what as the *ONNX*
# has more operators than the original *scikit-learn* pipelines.
# The graph at :ref:`l-plot-complex-pipeline-graph`
# helps up to find the outputs of both numerical
# and textual pipeline: *variable1*, *variable2*.
# Let's look into the numerical pipeline first.

num_onnx = select_model_inputs_outputs(model_onnx, 'variable1')
save_onnx_model(num_onnx, "pipeline_titanic_numerical.onnx")

################################
# Let's compute the numerical features.

sess = rt.InferenceSession("pipeline_titanic_numerical.onnx")
numX = sess.run(None, inputs)
print("numerical features", numX[0][:1])

###########################################
# We do the same for the textual features.

print(model_onnx)
text_onnx = select_model_inputs_outputs(model_onnx, 'variable2')
save_onnx_model(text_onnx, "pipeline_titanic_textual.onnx")
sess = rt.InferenceSession("pipeline_titanic_textual.onnx")
numT = sess.run(None, inputs)
print("textual features", numT[0][:1])

##################################
# Display the sub-ONNX graph
# ++++++++++++++++++++++++++
#
# Finally, let's see both subgraphs. First, numerical pipeline.

pydot_graph = GetPydotGraph(
    num_onnx.graph, name=num_onnx.graph.name, rankdir="TB",
    node_producer=GetOpNodeProducer(
        "docstring", color="yellow", fillcolor="yellow", style="filled"))
pydot_graph.write_dot("pipeline_titanic_num.dot")

os.system('dot -O -Gdpi=300 -Tpng pipeline_titanic_num.dot')

image = plt.imread("pipeline_titanic_num.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')

######################################
# Then textual pipeline.

pydot_graph = GetPydotGraph(
    text_onnx.graph, name=text_onnx.graph.name, rankdir="TB",
    node_producer=GetOpNodeProducer(
        "docstring", color="yellow", fillcolor="yellow", style="filled"))
pydot_graph.write_dot("pipeline_titanic_text.dot")

os.system('dot -O -Gdpi=300 -Tpng pipeline_titanic_text.dot')

image = plt.imread("pipeline_titanic_text.dot.png")
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
