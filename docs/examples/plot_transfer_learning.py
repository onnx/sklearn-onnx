# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-transfer-learning:

Transfer learning with ONNX
===========================


.. contents::
    :local:

Train a model
+++++++++++++

A very basic example using random forest and
the iris dataset.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)
print(clr)

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([1, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)

with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

###################################
# Compute ONNX prediction similarly as scikit-learn transformer
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from onnxruntime.sklapi import OnnxTransformer

with open("rf_iris.onnx", "rb") as f:
    content = f.read()

ot = OnnxTransformer(content, output_name="output_probability")
ot.fit(X_train, y_train)

print(ot.transform(X_test[:5]))

