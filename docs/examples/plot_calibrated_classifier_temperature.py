# SPDX-License-Identifier: Apache-2.0


"""
.. _l-calibrated-classifier-temperature:

Temperature scaling with CalibratedClassifierCV
===============================================

Temperature scaling is available in scikit-learn >= 1.8. It adjusts
class probabilities by scaling scores with one learned temperature
before applying softmax. This example shows how to calibrate a fitted
classifier, convert it to ONNX, and compare the predictions.

Train a classifier
++++++++++++++++++

The training, calibration, and test sets must be separate.
"""

import matplotlib.pyplot as plt
import numpy
from numpy.testing import assert_allclose, assert_array_equal
import onnx
import onnxruntime as rt
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import skl2onnx
from skl2onnx import to_onnx

iris = load_iris()
X, y = iris.data.astype(numpy.float32), iris.target
X_train, X_other, y_train, y_other = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_other, y_other, test_size=0.5, random_state=42, stratify=y_other
)
classifier = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(classifier)


###################################
# Calibrate on held-out data
# ++++++++++++++++++++++++++
#
# :class:`sklearn.frozen.FrozenEstimator` keeps the fitted classifier
# fixed while the calibrator learns the temperature on ``X_cal``.

calibrated = CalibratedClassifierCV(
    FrozenEstimator(classifier), method="temperature"
).fit(X_cal, y_cal)
print(calibrated)


###################################
# Convert the calibrated classifier
# +++++++++++++++++++++++++++++++++
#
# Disabling ZipMap returns the probabilities as a tensor rather than
# a list of dictionaries (see :ref:`l-rf-example-zipmap`).

onx = to_onnx(calibrated, X_test[:1], target_opset=18, options={"zipmap": False})
sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
pred_onx, proba_onx = sess.run(None, {"X": X_test})


###################################
# Compare the predictions
# +++++++++++++++++++++++
#
# The exported model should reproduce the calibrated classifier's
# labels and probabilities, within floating-point precision.

pred_skl = calibrated.predict(X_test)
proba_skl = calibrated.predict_proba(X_test)
assert_array_equal(pred_skl, pred_onx)
assert_allclose(proba_skl, proba_onx, rtol=1e-5, atol=1e-6)
assert_array_equal(classifier.predict(X_test), pred_onx)
print("scikit-learn:", proba_skl[:3])
print("onnxruntime:", proba_onx[:3])
print("Maximum probability difference:", numpy.max(numpy.abs(proba_skl - proba_onx)))


###################################
# Plot the probabilities
# ++++++++++++++++++++++
#
# The left plot shows how calibration changes the probabilities.
# The right plot compares the calibrated probabilities in scikit-learn
# and ONNX Runtime. Points on the diagonal indicate agreement.

proba_uncalibrated = classifier.predict_proba(X_test)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for k, label in enumerate(calibrated.classes_):
    name = iris.target_names[label]
    axes[0].scatter(proba_uncalibrated[:, k], proba_skl[:, k], label=name, alpha=0.7)
    axes[1].scatter(proba_skl[:, k], proba_onx[:, k], label=name, alpha=0.7)

for ax in axes:
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

axes[0].set_title("Temperature scaling")
axes[0].set_xlabel("Uncalibrated probability")
axes[0].set_ylabel("Calibrated probability")
axes[1].set_title("ONNX conversion")
axes[1].set_xlabel("scikit-learn probability")
axes[1].set_ylabel("ONNX Runtime probability")
fig.tight_layout()
plt.show()


#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
