# SPDX-License-Identifier: Apache-2.0

"""
Store arrays in one onnx graph
==============================

Once a model is converted it can be useful to store an
array as a constant in the graph an retrieve it through
an output. This allows the user to store training parameters
or other informations like a vocabulary.
Last sections shows how to remove an output or to promote
an intermediate result to an output.

.. contents::
    :local:

Train and convert a model
+++++++++++++++++++++++++

We download one model from the :epkg:`ONNX Zoo` but the model
could be trained and produced by another converter library.
"""
import pprint
import numpy
from onnx import load
from onnxruntime import InferenceSession
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx
from skl2onnx.helpers.onnx_helper import (
    add_output_initializer, select_model_inputs_outputs)


data = load_iris()
X, y = data.data.astype(numpy.float32), data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression(penalty='elasticnet', C=2.,
                           solver='saga', l1_ratio=0.5)
model.fit(X_train, y_train)

onx = to_onnx(model, X_train[:1], target_opset=12,
              options={'zipmap': False})

########################################
# Add training parameter
# ++++++++++++++++++++++
#

new_onx = add_output_initializer(
    onx,
    ['C', 'l1_ratio'],
    [numpy.array([model.C]), numpy.array([model.l1_ratio])])

########################################
# Inference
# +++++++++

sess = InferenceSession(new_onx.SerializeToString())
print("output names:", [o.name for o in sess.get_outputs()])
res = sess.run(None, {'X': X_test[:2]})
print("outputs")
pprint.pprint(res)


#######################################
# The major draw back of this solution is increase the prediction
# time as onnxruntime copies the constants for every prediction.
# It is possible either to store those constant in a separate ONNX graph
# or to removes them.
#
# Select outputs
# ++++++++++++++
#
# Next function removes unneeded outputs from a model,
# not only the constants. Next model only keeps the probabilities.

simple_onx = select_model_inputs_outputs(new_onx, ['probabilities'])

sess = InferenceSession(simple_onx.SerializeToString())
print("output names:", [o.name for o in sess.get_outputs()])
res = sess.run(None, {'X': X_test[:2]})
print("outputs")
pprint.pprint(res)

# Function *select_model_inputs_outputs* add also promote an intermediate
# result to an output.
#
#####################################
# This example only uses ONNX graph in memory and never saves or loads a
# model. This can be done by using the following snippets of code.
#
# Save a model
# ++++++++++++

with open("simplified_model.onnx", "wb") as f:
    f.write(simple_onx.SerializeToString())

###################################
# Load a model
# ++++++++++++


model = load("simplified_model.onnx", "wb")

sess = InferenceSession(model.SerializeToString())
print("output names:", [o.name for o in sess.get_outputs()])
res = sess.run(None, {'X': X_test[:2]})
print("outputs")
pprint.pprint(res)
