# SPDX-License-Identifier: Apache-2.0


"""
.. _l-rf-iris-example:

Train, convert and predict a model
==================================

Train and deploy a model usually involves the
three following steps:

* train a pipeline with *scikit-learn*,
* convert it into *ONNX* with *sklearn-onnx*,
* predict with *onnxruntime*.

.. contents::
    :local:

Train a model
+++++++++++++

A very basic example using random forest and
the iris dataset.
"""

import skl2onnx
import onnx
import sklearn
from sklearn.linear_model import LogisticRegression
import numpy
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
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

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type,
                      target_opset=12)

with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

###################################
# Compute the prediction with ONNX Runtime
# ++++++++++++++++++++++++++++++++++++++++
sess = rt.InferenceSession("rf_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

#######################################
# Full example with a logistic regression

clr = LogisticRegression()
clr.fit(X_train, y_train)
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onx = convert_sklearn(clr, initial_types=initial_type,
                      target_opset=12)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("logreg_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name],
                    {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)


#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
