# SPDX-License-Identifier: Apache-2.0


"""
.. _l-example-logging:

Logging, verbose
================

The conversion of a pipeline fails if it contains an object without any
associated converter. It may also fails if one of the object is mapped
by a custom converter. If the error message is not explicit enough,
it is possible to enable logging.


.. contents::
    :local:

Train a model
+++++++++++++

A very basic example using random forest and
the iris dataset.
"""

import logging
import numpy
import onnx
import onnxruntime as rt
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import skl2onnx

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = DecisionTreeClassifier()
clr.fit(X_train, y_train)
print(clr)

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type,
                      target_opset=12)


sess = rt.InferenceSession(onx.SerializeToString())
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name],
                    {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

########################################
# Conversion with parameter verbose
# +++++++++++++++++++++++++++++++++
#
# verbose is a parameter which prints messages on the standard output.
# It tells which converter is called. `verbose=1` usually means what *skl2onnx*
# is doing to convert a pipeline. `verbose=2+`
# is reserved for information within converters.

convert_sklearn(clr, initial_types=initial_type, target_opset=12, verbose=1)

########################################
# Conversion with logging
# +++++++++++++++++++++++
#
# This is very detailed logging. It which operators or variables
# (output of converters) is processed, which node is created...
# This information may be useful when a custom converter is being
# implemented.

logger = logging.getLogger('skl2onnx')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

convert_sklearn(clr, initial_types=initial_type, target_opset=12)

###########################
# And to disable it.

logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

convert_sklearn(clr, initial_types=initial_type, target_opset=12)


#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
