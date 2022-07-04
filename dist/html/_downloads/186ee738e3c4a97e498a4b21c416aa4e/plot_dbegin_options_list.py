# SPDX-License-Identifier: Apache-2.0

"""
Black list operators when converting
====================================

.. index:: black list, white list

Some runtimes do not implement a runtime for every
available operator in ONNX. The converter does not know
that but it is possible to black some operators. Most of
the converters do not change their behaviour, they fail
if they use a black listed operator, a couple of them
produces a different ONNX graph.

.. contents::
    :local:

GaussianMixture
+++++++++++++++

The first converter to change its behaviour depending on a black list
of operators is for model *GaussianMixture*.
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
from timeit import timeit
import numpy
from onnxruntime import InferenceSession
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx

data = load_iris()
X_train, X_test = train_test_split(data.data)
model = GaussianMixture()
model.fit(X_train)

###################################
# Default conversion
# ++++++++++++++++++

model_onnx = to_onnx(
    model, X_train[:1].astype(numpy.float32),
    options={id(model): {'score_samples': True}},
    target_opset=12)
sess = InferenceSession(model_onnx.SerializeToString())

xt = X_test[:5].astype(numpy.float32)
print(model.score_samples(xt))
print(sess.run(None, {'X': xt})[2])


##################################
# Display the ONNX graph.


oinf = OnnxInference(model_onnx)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

###################################
# Conversion without ReduceLogSumExp
# ++++++++++++++++++++++++++++++++++
#
# Parameter *black_op* is used to tell the converter
# not to use this operator. Let's see what the converter
# produces in that case.

model_onnx2 = to_onnx(
    model, X_train[:1].astype(numpy.float32),
    options={id(model): {'score_samples': True}},
    black_op={'ReduceLogSumExp'},
    target_opset=12)
sess2 = InferenceSession(model_onnx2.SerializeToString())

xt = X_test[:5].astype(numpy.float32)
print(model.score_samples(xt))
print(sess2.run(None, {'X': xt})[2])

##################################
# Display the ONNX graph.

oinf = OnnxInference(model_onnx2)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


#######################################
# Processing time
# +++++++++++++++

print(timeit(stmt="sess.run(None, {'X': xt})",
             number=10000, globals={'sess': sess, 'xt': xt}))

print(timeit(stmt="sess2.run(None, {'X': xt})",
             number=10000, globals={'sess2': sess2, 'xt': xt}))

#################################
# The model using ReduceLogSumExp is much faster.

##########################################
# If the converter cannot convert without...
# ++++++++++++++++++++++++++++++++++++++++++
#
# Many converters do not consider the white and black lists
# of operators. If a converter fails to convert without using
# a blacklisted operator (or only whitelisted operators),
# *skl2onnx* raises an error.

try:
    to_onnx(
        model, X_train[:1].astype(numpy.float32),
        options={id(model): {'score_samples': True}},
        black_op={'ReduceLogSumExp', 'Add'},
        target_opset=12)
except RuntimeError as e:
    print('Error:', e)
