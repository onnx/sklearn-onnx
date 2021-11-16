# SPDX-License-Identifier: Apache-2.0


"""
.. _l-tutorial-example-zipmap:

Choose appropriate output of a classifier
=========================================

A scikit-learn classifier usually returns a matrix of probabilities.
By default, *sklearn-onnx* converts that matrix
into a list of dictionaries where each probabily is mapped
to its class id or name. That mechanism retains the class names
but is slower. Let's see what other options are available.

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
from skl2onnx import to_onnx
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(numpy.float32)
y = y * 2 + 10  # to get labels different from [0, 1, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression(max_iter=500)
clr.fit(X_train, y_train)
print(clr)

onx = to_onnx(clr, X_train, target_opset=12)

############################
# Default behaviour: zipmap=True
# ++++++++++++++++++++++++++++++
#
# The output type for the probabilities is a list of
# dictionaries.

sess = rt.InferenceSession(onx.SerializeToString())
res = sess.run(None, {'X': X_test})
print(res[1][:2])
print("probabilities type:", type(res[1]))
print("type for the first observations:", type(res[1][0]))

###################################
# Option zipmap=False
# +++++++++++++++++++
#
# Probabilities are now a matrix.

initial_type = [('float_input', FloatTensorType([None, 4]))]
options = {id(clr): {'zipmap': False}}
onx2 = to_onnx(clr, X_train, options=options, target_opset=12)

sess2 = rt.InferenceSession(onx2.SerializeToString())
res2 = sess2.run(None, {'X': X_test})
print(res2[1][:2])
print("probabilities type:", type(res2[1]))
print("type for the first observations:", type(res2[1][0]))

###################################
# Option zipmap='columns'
# +++++++++++++++++++++++
#
# This options removes the final operator ZipMap and splits
# the probabilities into columns. The final model produces
# one output for the label, and one output per class.

options = {id(clr): {'zipmap': 'columns'}}
onx3 = to_onnx(clr, X_train, options=options, target_opset=12)

sess3 = rt.InferenceSession(onx3.SerializeToString())
res3 = sess3.run(None, {'X': X_test})
for i, out in enumerate(sess3.get_outputs()):
    print("output: '{}' shape={} values={}...".format(
        out.name, res3[i].shape, res3[i][:2]))


###################################
# Let's compare prediction time
# +++++++++++++++++++++++++++++

print("Average time with ZipMap:")
print(sum(repeat(lambda: sess.run(None, {'X': X_test}),
                 number=100, repeat=10)) / 10)

print("Average time without ZipMap:")
print(sum(repeat(lambda: sess2.run(None, {'X': X_test}),
                 number=100, repeat=10)) / 10)

print("Average time without ZipMap but with columns:")
print(sum(repeat(lambda: sess3.run(None, {'X': X_test}),
                 number=100, repeat=10)) / 10)

# The prediction is much faster without ZipMap
# on this example.
# The optimisation is even faster when the classes
# are described with strings and not integers
# as the final result (list of dictionaries) may copy
# many times the same information with onnxruntime.

#######################################
# Option zimpap=False and output_class_labels=True
# ++++++++++++++++++++++++++++++++++++++++++++++++
#
# Option `zipmap=False` seems a better choice because it is
# much faster but labels are lost in the process. Option
# `output_class_labels` can be used to expose the labels
# as a third output.

initial_type = [('float_input', FloatTensorType([None, 4]))]
options = {id(clr): {'zipmap': False, 'output_class_labels': True}}
onx4 = to_onnx(clr, X_train, options=options, target_opset=12)

sess4 = rt.InferenceSession(onx4.SerializeToString())
res4 = sess4.run(None, {'X': X_test})
print(res4[1][:2])
print("probabilities type:", type(res4[1]))
print("class labels:", res4[2])

###########################################
# Processing time.

print("Average time without ZipMap but with output_class_labels:")
print(sum(repeat(lambda: sess4.run(None, {'X': X_test}),
                 number=100, repeat=10)) / 10)

###########################################
# MultiOutputClassifier
# +++++++++++++++++++++
#
# This model is equivalent to several classifiers, one for every label
# to predict. Instead of returning a matrix of probabilities, it returns
# a sequence of matrices. Let's first modify the labels to get
# a problem for a MultiOutputClassifier.

y = numpy.vstack([y, y + 100]).T
y[::5, 1] = 1000  # Let's a fourth class.
print(y[:5])

########################################
# Let's train a MultiOutputClassifier.

X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = MultiOutputClassifier(LogisticRegression(max_iter=500))
clr.fit(X_train, y_train)
print(clr)

onx5 = to_onnx(clr, X_train, target_opset=12)

sess5 = rt.InferenceSession(onx5.SerializeToString())
res5 = sess5.run(None, {'X': X_test[:3]})
print(res5)

########################################
# Option zipmap is ignored. Labels are missing but they can be
# added back as a third output.

onx6 = to_onnx(clr, X_train, target_opset=12,
               options={'zipmap': False, 'output_class_labels': True})

sess6 = rt.InferenceSession(onx6.SerializeToString())
res6 = sess6.run(None, {'X': X_test[:3]})
print("predicted labels", res6[0])
print("predicted probabilies", res6[1])
print("class labels", res6[2])


#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
