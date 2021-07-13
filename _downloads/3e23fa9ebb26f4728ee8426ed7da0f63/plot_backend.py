# SPDX-License-Identifier: Apache-2.0


"""

.. _l-example-backend-api:

ONNX Runtime Backend for ONNX
=============================

.. index:: backend

*ONNX Runtime* extends the
`onnx backend API <https://github.com/onnx/onnx/blob/master/docs/
ImplementingAnOnnxBackend.md>`_
to run predictions using this runtime.
Let's use the API to compute the prediction
of a simple logistic regression model.
"""
import skl2onnx
import onnxruntime
import onnx
import sklearn
import numpy
from onnxruntime import get_device
import numpy as np
from onnxruntime import datasets
import onnxruntime.backend as backend
from onnx import load

name = datasets.get_example("logreg_iris.onnx")
model = load(name)

rep = backend.prepare(model, 'CPU')
x = np.array([[-1.0, -2.0, 5.0, 6.0],
              [-1.0, -2.0, -3.0, -4.0],
              [-1.0, -2.0, 7.0, 8.0]],
             dtype=np.float32)
label, proba = rep.run(x)
print("label={}".format(label))
print("probabilities={}".format(proba))

########################################
# The device depends on how the package was compiled,
# GPU or CPU.
print(get_device())

########################################
# The backend can also directly load the model
# without using *onnx*.

rep = backend.prepare(name, 'CPU')
x = np.array([[-1.0, -2.0, -3.0, -4.0],
              [-1.0, -2.0, -3.0, -4.0],
              [-1.0, -2.0, -3.0, -4.0]],
             dtype=np.float32)
label, proba = rep.run(x)
print("label={}".format(label))
print("probabilities={}".format(proba))

#######################################
# The backend API is implemented by other frameworks
# and makes it easier to switch between multiple runtimes
# with the same API.

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
