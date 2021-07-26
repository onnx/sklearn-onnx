# SPDX-License-Identifier: Apache-2.0


"""
.. _l-rf-example-decision-function:

Probabilities or raw scores
===========================

A classifier usually returns a matrix of probabilities.
By default, *sklearn-onnx* creates an ONNX graph
which returns probabilities but it may skip that
step and return raw scores if the model implements
the method *decision_function*. Option ``'raw_scores'``
is used to change the default behaviour. Let's see
that on a simple example.

.. contents::
    :local:

Train a model and convert it
++++++++++++++++++++++++++++

"""
import numpy
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import onnxruntime as rt
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression(max_iter=500)
clr.fit(X_train, y_train)
print(clr)

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type,
                      target_opset=12)

############################
# Output type
# +++++++++++
#
# Let's confirm the output type of the probabilities
# is a list of dictionaries with onnxruntime.

sess = rt.InferenceSession(onx.SerializeToString())
res = sess.run(None, {'float_input': X_test.astype(numpy.float32)})
print("skl", clr.predict_proba(X_test[:1]))
print("onnx", res[1][:2])

###################################
# Raw scores and decision_function
# ++++++++++++++++++++++++++++++++
#

initial_type = [('float_input', FloatTensorType([None, 4]))]
options = {id(clr): {'raw_scores': True}}
onx2 = convert_sklearn(clr, initial_types=initial_type, options=options,
                       target_opset=12)

sess2 = rt.InferenceSession(onx2.SerializeToString())
res2 = sess2.run(None, {'float_input': X_test.astype(numpy.float32)})
print("skl", clr.decision_function(X_test[:1]))
print("onnx", res2[1][:2])

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
