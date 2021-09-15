# SPDX-License-Identifier: Apache-2.0


"""
.. _example-lightgbm:

Convert a pipeline with a LightGBM classifier
=============================================

.. index:: LightGBM

:epkg:`sklearn-onnx` only converts :epkg:`scikit-learn` models into *ONNX*
but many libraries implement :epkg:`scikit-learn` API so that their models
can be included in a :epkg:`scikit-learn` pipeline. This example considers
a pipeline including a :epkg:`LightGBM` model. :epkg:`sklearn-onnx` can convert
the whole pipeline as long as it knows the converter associated to
a *LGBMClassifier*. Let's see how to do it.

.. contents::
    :local:

Train a LightGBM classifier
+++++++++++++++++++++++++++
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
import onnxruntime as rt
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm  # noqa
from skl2onnx.common.data_types import FloatTensorType
import numpy
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

data = load_iris()
X = data.data[:, :2]
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

pipe = Pipeline([('scaler', StandardScaler()),
                 ('lgbm', LGBMClassifier(n_estimators=3))])
pipe.fit(X, y)

######################################
# Register the converter for LGBMClassifier
# +++++++++++++++++++++++++++++++++++++++++
#
# The converter is implemented in :epkg:`onnxmltools`:
# `onnxmltools...LightGbm.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# lightgbm/operator_converters/LightGbm.py>`_.
# and the shape calculator:
# `onnxmltools...Classifier.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# lightgbm/shape_calculators/Classifier.py>`_.

update_registered_converter(
    LGBMClassifier, 'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes, convert_lightgbm,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

##################################
# Convert again
# +++++++++++++

model_onnx = convert_sklearn(
    pipe, 'pipeline_lightgbm',
    [('input', FloatTensorType([None, 2]))],
    target_opset=12)

# And save.
with open("pipeline_lightgbm.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

###########################
# Compare the predictions
# +++++++++++++++++++++++
#
# Predictions with LightGbm.

print("predict", pipe.predict(X[:5]))
print("predict_proba", pipe.predict_proba(X[:1]))

##########################
# Predictions with onnxruntime.

sess = rt.InferenceSession("pipeline_lightgbm.onnx")

pred_onx = sess.run(None, {"input": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])

#############################
# Final graph
# +++++++++++


oinf = OnnxInference(model_onnx)
ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
