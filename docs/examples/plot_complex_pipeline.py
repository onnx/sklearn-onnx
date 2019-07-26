# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _example-complex-pipeline:

Convert a pipeline with ColumnTransformer
=========================================

*scikit-learn* recently shipped
`ColumnTransformer <https://scikit-learn.org/stable/modules/
generated/sklearn.compose.ColumnTransformer.html>`_
which lets the user define complex pipeline where each
column may be preprocessed with a different transformer.
*sklearn-onnx* still works in this case as shown in Section
:ref:`l-complex-pipeline`.


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
import onnxruntime
import onnx
import sklearn
import matplotlib.pyplot as plt
import os
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import numpy
import onnxruntime as rt
from skl2onnx import convert_sklearn
import pprint
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.data_types import Int64TensorType
import pandas as pd
import numpy as np
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

# SimpleImputer on string is not available for
# string in ONNX-ML specifications.
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
    model_onnx = convert_sklearn(clf, 'pipeline_titanic', inputs)
except Exception as e:
    print(e)

#################################
# Predictions are more efficient if the graph is small.
# That's why the converter checks that there is no unused input.
# They need to be removed from the graph inputs.

to_drop = {'parch', 'sibsp', 'cabin', 'ticket',
           'name', 'body', 'home.dest', 'boat'}
inputs = convert_dataframe_schema(X_train, to_drop)
try:
    model_onnx = convert_sklearn(clf, 'pipeline_titanic', inputs)
except Exception as e:
    print(e)

#################################
# *scikit-learn* does implicit conversions when it can.
# *sklearn-onnx* does not. The ONNX version of *OneHotEncoder*
# must be applied on columns of the same type.

X_train['pclass'] = X_train['pclass'].astype(str)
X_test['pclass'] = X_test['pclass'].astype(str)
inputs = convert_dataframe_schema(X_train, to_drop)

model_onnx = convert_sklearn(clf, 'pipeline_titanic', inputs)


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

##################################
# .. _l-plot-complex-pipeline-graph:
#
# Display the ONNX graph
# ++++++++++++++++++++++
#
# Finally, let's see the graph converted with *sklearn-onnx*.

pydot_graph = GetPydotGraph(model_onnx.graph, name=model_onnx.graph.name,
                            rankdir="TB",
                            node_producer=GetOpNodeProducer("docstring",
                                                            color="yellow",
                                                            fillcolor="yellow",
                                                            style="filled"))
pydot_graph.write_dot("pipeline_titanic.dot")

os.system('dot -O -Gdpi=300 -Tpng pipeline_titanic.dot')

image = plt.imread("pipeline_titanic.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
