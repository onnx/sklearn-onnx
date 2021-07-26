# SPDX-License-Identifier: Apache-2.0


"""
Investigate a pipeline
======================

The following example shows how to look into a converted
models and easily find errors at every step of the pipeline.

.. contents::
    :local:

Create a pipeline
+++++++++++++++++

We reuse the pipeline implemented in example
`Pipelining: chaining a PCA and a logistic regression
<https://scikit-learn.org/stable/auto_examples/
compose/plot_digits_pipe.html>`_.
There is one change because
`ONNX-ML Imputer
<https://github.com/onnx/onnx/blob/master/docs/
Operators-ml.md#ai.onnx.ml.Imputer>`_
does not handle string type. This cannot be part of the final ONNX pipeline
and must be removed. Look for comment starting with ``---`` below.
"""
import skl2onnx
import onnx
import sklearn
import numpy
import pickle
from skl2onnx.helpers import collect_intermediate_steps
import onnxruntime as rt
from onnxconverter_common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[('pca', PCA()),
                       ('logistic', LogisticRegression())])

digits = datasets.load_digits()
X_digits = digits.data[:1000]
y_digits = digits.target[:1000]

pipe.fit(X_digits, y_digits)

###############################################
# Conversion to ONNX
# ++++++++++++++++++


initial_types = [('input', FloatTensorType((None, X_digits.shape[1])))]
model_onnx = convert_sklearn(pipe, initial_types=initial_types,
                             target_opset=12)

sess = rt.InferenceSession(model_onnx.SerializeToString())
print("skl predict_proba")
print(pipe.predict_proba(X_digits[:2]))
onx_pred = sess.run(None, {'input': X_digits[:2].astype(np.float32)})[1]
df = pd.DataFrame(onx_pred)
print("onnx predict_proba")
print(df.values)

###############################################
# Intermediate steps
# ++++++++++++++++++
#
# Let's imagine the final output is wrong and we need
# to look into each component of the pipeline which one
# is failing. The following method modifies the scikit-learn
# pipeline to steal the intermediate outputs and produces
# an smaller ONNX graph for every operator.


steps = collect_intermediate_steps(pipe, "pipeline",
                                   initial_types)

assert len(steps) == 2

pipe.predict_proba(X_digits[:2])

for i, step in enumerate(steps):
    onnx_step = step['onnx_step']
    sess = rt.InferenceSession(onnx_step.SerializeToString())
    onnx_outputs = sess.run(None, {'input': X_digits[:2].astype(np.float32)})
    skl_outputs = step['model']._debug.outputs
    print("step 1", type(step['model']))
    print("skl outputs")
    print(skl_outputs)
    print("onnx outputs")
    print(onnx_outputs)

########################################
# Pickle
# ++++++
#
# Each steps is a separate model in the pipeline.
# It can be pickle independetly from the others.
# Attribute *_debug* contains all the information
# needed to *replay* the prediction of the model.

to_save = {
    'model': steps[1]['model'],
    'data_input': steps[1]['model']._debug.inputs,
    'data_output': steps[1]['model']._debug.outputs,
    'inputs': steps[1]['inputs'],
    'outputs': steps[1]['outputs'],
}
del steps[1]['model']._debug

with open('classifier.pkl', 'wb') as f:
    pickle.dump(to_save, f)

with open('classifier.pkl', 'rb') as f:
    restored = pickle.load(f)

print(restored['model'].predict_proba(restored['data_input']['predict_proba']))

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
