# SPDX-License-Identifier: Apache-2.0


"""
.. _l-black-op:

Convert a model with a reduced list of operators
================================================

Some runtime dedicated to onnx do not implement all the
operators and a converted model may not run if one of them
is missing from the list of available operators.
Some converters may convert a model in different ways
if the users wants to blacklist some operators.

GaussianMixture
+++++++++++++++

The first converter to change its behaviour depending on a black list
of operators is for model *GaussianMixture*.
"""

import onnxruntime
import onnx
import numpy
import os
from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt
from onnx_array_api.plotting.dot_plot import to_dot
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
    model,
    X_train[:1].astype(np.float32),
    options={id(model): {"score_samples": True}},
    target_opset=12,
)
sess = InferenceSession(
    model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
)

xt = X_test[:5].astype(np.float32)
print(model.score_samples(xt))
print(sess.run(None, {"X": xt})[2])


##################################
# Display the ONNX graph.

with open("mixture.dot", "w", encoding="utf-8") as f:
    f.write(to_dot(model_onnx))

os.system("dot -O -Gdpi=300 -Tpng mixture.dot")

image = plt.imread("mixture.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis("off")


###################################
# Conversion without ReduceLogSumExp
# ++++++++++++++++++++++++++++++++++
#
# Parameter *black_op* is used to tell the converter
# not to use this operator. Let's see what the converter
# produces in that case.

model_onnx2 = to_onnx(
    model,
    X_train[:1].astype(np.float32),
    options={id(model): {"score_samples": True}},
    black_op={"ReduceLogSumExp"},
    target_opset=12,
)
sess2 = InferenceSession(
    model_onnx2.SerializeToString(), providers=["CPUExecutionProvider"]
)

xt = X_test[:5].astype(np.float32)
print(model.score_samples(xt))
print(sess2.run(None, {"X": xt})[2])

##################################
# Display the ONNX graph.

with open("mixture2.dot", "w", encoding="utf-8") as f:
    f.write(to_dot(model_onnx2))

os.system("dot -O -Gdpi=300 -Tpng mixture2.dot")

image = plt.imread("mixture2.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis("off")


#######################################
# Processing time
# +++++++++++++++

print(
    timeit(
        stmt="sess.run(None, {'X': xt})", number=10000, globals={"sess": sess, "xt": xt}
    )
)

print(
    timeit(
        stmt="sess2.run(None, {'X': xt})",
        number=10000,
        globals={"sess2": sess2, "xt": xt},
    )
)

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
        model,
        X_train[:1].astype(np.float32),
        options={id(model): {"score_samples": True}},
        black_op={"ReduceLogSumExp", "Add"},
        target_opset=12,
    )
except RuntimeError as e:
    print("Error:", e)


#################################
# **Versions used for this example**

import sklearn

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
import skl2onnx

print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
