# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-gpr-example:

Discrepencies with GaussianProcessorRegressor: sue of double
============================================================

The `GaussianProcessRegressor
<https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.
GaussianProcessRegressor.html>`_ involves
many matrix operations which may requires double
precisions. *sklearn-onnx* is using single floats by default
but for this particular model, it is better to use double.
Let's see how to create an ONNX file using doubles.

.. contents::
    :local:

Train a model
+++++++++++++

A very basic example using *GaussianProcessRegressor*
on the Boston dataset.
"""

import skl2onnx
import onnxruntime
import onnx
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
import numpy
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import convert_sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

bost = load_boston()
X, y = bost.data, bost.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
gpr = GaussianProcessRegressor(DotProduct(), alpha=1.)
gpr.fit(X_train, y_train)
print(gpr)

###########################
# First attempt to convert a model into ONNX
# ++++++++++++++++++++++++++++++++++++++++++
#
# The documentation suggests the following way to
# convert a model into ONNX.

initial_type = [('X', FloatTensorType([1, X_train.shape[1]]))]
onx = convert_sklearn(gpr, initial_types=initial_type)

sess = rt.InferenceSession(onx.SerializeToString())
try:
    pred_onx = sess.run(
        None, {'X': X_test.astype(numpy.float32)})[0]
except RuntimeError as e:
    print(str(e))

###########################
# Second attempt: variable dimensions
# +++++++++++++++++++++++++++++++++++
#
# Unfortunately, even though the conversion
# went well, the runtime fails to compute the prediction.
# The previous snippet of code imposes fixed dimension
# on the input and therefore let the runtime assume
# every node output has outputs with fixed dimensions
# And that's not the case for this model.
# We need to disable these checkings by replacing
# the fixed dimensions by variables dimensions
# or string (see next line).

initial_type = [('X', FloatTensorType(['d1', 'd2']))]
onx = convert_sklearn(gpr, initial_types=initial_type)

sess = rt.InferenceSession(onx.SerializeToString())
pred_onx = sess.run(
    None, {'X': X_test.astype(numpy.float32)})[0]

pred_skl = gpr.predict(X_test)
print(pred_skl[:10])
print(pred_onx[0, :10])

###################################
# The differences seems quite important.
# Let's confirm that by looking at the biggest
# differences.

diff = numpy.sort(numpy.abs(numpy.squeeze(pred_skl) -
                            numpy.squeeze(pred_onx)))[-5:]
print(diff)
print('min(Y)-max(Y):', min(y_test), max(y_test))

###########################
# Third attempt: use of double
# ++++++++++++++++++++++++++++
#
# The model uses a couple of matrix computations
# and matrices have coefficients with very different
# order of magnitude. It is difficult to approximate
# the prediction made with scikit-learn if the converted
# model sticks to float. Double precision is needed.
#
# The previous code requires two changes. The first
# one indicates that inputs are now of type
# ``DoubleTensorType``. The second change
# is the extra parameter ``dtype=numpy.float64``
# tells the conversion function that every real
# constant matrix such as the trained coefficients
# will be dumped as doubles and not as floats anymore.

initial_type = [('X', DoubleTensorType(['d1', 'd2']))]
onx64 = convert_sklearn(gpr, initial_types=initial_type,
                        dtype=numpy.float64)

sess64 = rt.InferenceSession(onx64.SerializeToString())
pred_onx64 = sess64.run(None, {'X': X_test})[0]

print(pred_onx64[0, :10])

################################
# The new differences look much better.

diff = numpy.sort(numpy.abs(numpy.squeeze(pred_skl) -
                            numpy.squeeze(pred_onx64)))[-5:]
print(diff)
print('min(Y)-max(Y):', min(y_test), max(y_test))

####################################
# Size increase
# +++++++++++++
#
# As a result, the ONNX model is almost twice bigger
# because every coefficient is stored as double and
# and not as floats anymore.

size32 = len(onx.SerializeToString())
size64 = len(onx64.SerializeToString())
print("ONNX with floats:", size32)
print("ONNX with doubles:", size64)

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
