# SPDX-License-Identifier: Apache-2.0

"""
Modify the ONNX graph
=====================

This example shows how to change the default ONNX graph such as
renaming the inputs or outputs names.

.. contents::
    :local:

Basic example
+++++++++++++

"""
import numpy
from onnxruntime import InferenceSession
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx import to_onnx

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(numpy.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = LogisticRegression(solver="liblinear")
clr.fit(X_train, y_train)


onx = to_onnx(clr, X, options={'zipmap': False})

sess = InferenceSession(onx.SerializeToString())
input_names = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]
print("inputs=%r, outputs=%r" % (input_names, output_names))
print(sess.run(None, {input_names[0]: X_test[:2]}))


####################################
# Changes the input names
# +++++++++++++++++++++++
#
# It is possible to change the input name by using the
# parameter *initial_types*. However, the user must specify the input
# types as well.

onx = to_onnx(clr, X, options={'zipmap': False},
              initial_types=[('X56', FloatTensorType([None, X.shape[1]]))])

sess = InferenceSession(onx.SerializeToString())
input_names = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]
print("inputs=%r, outputs=%r" % (input_names, output_names))
print(sess.run(None, {input_names[0]: X_test[:2]}))


####################################
# Changes the output names
# ++++++++++++++++++++++++
#
# It is possible to change the input name by using the
# parameter *final_types*.

onx = to_onnx(clr, X, options={'zipmap': False},
              final_types=[('L', Int64TensorType([None])),
                           ('P', FloatTensorType([None, 3]))])

sess = InferenceSession(onx.SerializeToString())
input_names = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]
print("inputs=%r, outputs=%r" % (input_names, output_names))
print(sess.run(None, {input_names[0]: X_test[:2]}))

####################################
# Renaming intermediate results
# +++++++++++++++++++++++++++++
#
# It is possible to rename intermediate results by using a prefix
# or by using a function. The result will be post-processed in order
# to unique names. It does not impact the graph inputs or outputs.


def rename_results(proposed_name, existing_names):
    result = "_" + proposed_name.upper()
    while result in existing_names:
        result += "A"
    print("changed %r into %r." % (proposed_name, result))
    return result


onx = to_onnx(clr, X, options={'zipmap': False},
              naming=rename_results)

sess = InferenceSession(onx.SerializeToString())
input_names = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]
print("inputs=%r, outputs=%r" % (input_names, output_names))
print(sess.run(None, {input_names[0]: X_test[:2]}))
