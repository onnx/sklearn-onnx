# SPDX-License-Identifier: Apache-2.0


"""
.. _l-rf-example-zipmap:

Probabilities as a vector or as a ZipMap
========================================

A classifier usually returns a matrix of probabilities.
By default, *sklearn-onnx* converts that matrix
into a list of dictionaries where each probabily is mapped
to its class id or name. That mechanism retains the class names.
This conversion increases the prediction time and is not
always needed. Let's see how to deactivate this behaviour
on the Iris example.

.. contents::
    :local:

Train a model and convert it
++++++++++++++++++++++++++++

"""
from timeit import repeat
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
print(res[1][:2])
print("probabilities type:", type(res[1]))
print("type for the first observations:", type(res[1][0]))

###################################
# Without ZipMap
# ++++++++++++++
#
# Let's remove the ZipMap operator.

initial_type = [('float_input', FloatTensorType([None, 4]))]
options = {id(clr): {'zipmap': False}}
onx2 = convert_sklearn(clr, initial_types=initial_type, options=options,
                       target_opset=12)

sess2 = rt.InferenceSession(onx2.SerializeToString())
res2 = sess2.run(None, {'float_input': X_test.astype(numpy.float32)})
print(res2[1][:2])
print("probabilities type:", type(res2[1]))
print("type for the first observations:", type(res2[1][0]))

###################################
# One output per class
# ++++++++++++++++++++
#
# This options removes the final operator ZipMap and splits
# the probabilities into columns. The final model produces
# one output for the label, and one output per class.

options = {id(clr): {'zipmap': 'columns'}}
onx3 = convert_sklearn(clr, initial_types=initial_type, options=options,
                       target_opset=12)

sess3 = rt.InferenceSession(onx3.SerializeToString())
res3 = sess3.run(None, {'float_input': X_test.astype(numpy.float32)})
for i, out in enumerate(sess3.get_outputs()):
    print("output: '{}' shape={} values={}...".format(
        out.name, res3[i].shape, res3[i][:2]))


###################################
# Let's compare prediction time
# +++++++++++++++++++++++++++++

X32 = X_test.astype(numpy.float32)

print("Time with ZipMap:")
print(repeat(lambda: sess.run(None, {'float_input': X32}),
             number=100, repeat=10))

print("Time without ZipMap:")
print(repeat(lambda: sess2.run(None, {'float_input': X32}),
             number=100, repeat=10))

print("Time without ZipMap but with columns:")
print(repeat(lambda: sess3.run(None, {'float_input': X32}),
             number=100, repeat=10))

# The prediction is much faster without ZipMap
# on this example.
# The optimisation is even faster when the classes
# are described with strings and not integers
# as the final result (list of dictionaries) may copy
# many times the same information with onnxruntime.

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
