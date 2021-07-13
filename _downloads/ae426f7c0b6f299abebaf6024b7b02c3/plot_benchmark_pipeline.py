# SPDX-License-Identifier: Apache-2.0


"""
Benchmark a pipeline
====================

The following example checks up on every step in a pipeline,
compares and benchmarks the predictions.

.. contents::
    :local:

Create a pipeline
+++++++++++++++++

We reuse the pipeline implemented in example
`Pipelining: chaining a PCA and a logistic regression
<https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html>`_.
There is one change because
`ONNX-ML Imputer <https://github.com/onnx/onnx/blob/master/
docs/Operators-ml.md#ai.onnx.ml.Imputer>`_
does not handle string type. This cannot be part of the final ONNX pipeline
and must be removed. Look for comment starting with ``---`` below.
"""
import skl2onnx
import onnx
import sklearn
import numpy
from skl2onnx.helpers import collect_intermediate_steps
from timeit import timeit
from skl2onnx.helpers import compare_objects
import onnxruntime as rt
from onnxconverter_common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logistic = LogisticRegression()
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

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
# Comparing outputs
# +++++++++++++++++

compare_objects(pipe.predict_proba(X_digits[:2]), onx_pred)
# No exception so they are the same.

###############################################
# Benchmarks
# ++++++++++

print("scikit-learn")
print(timeit("pipe.predict_proba(X_digits[:1])",
             number=10000, globals=globals()))
print("onnxruntime")
print(timeit("sess.run(None, {'input': X_digits[:1].astype(np.float32)})[1]",
             number=10000, globals=globals()))

###############################################
# Intermediate steps
# ++++++++++++++++++
#
# Let's imagine the final output is wrong and we need
# to look into each component of the pipeline which one
# is failing. The following method modifies the scikit-learn
# pipeline to steal the intermediate outputs and produces
# an smaller ONNX graph for every operator.


steps = collect_intermediate_steps(
    pipe, "pipeline", initial_types)

assert len(steps) == 2

pipe.predict_proba(X_digits[:2])

for i, step in enumerate(steps):
    onnx_step = step['onnx_step']
    sess = rt.InferenceSession(onnx_step.SerializeToString())
    onnx_outputs = sess.run(None, {'input': X_digits[:2].astype(np.float32)})
    skl_outputs = step['model']._debug.outputs
    if 'transform' in skl_outputs:
        compare_objects(skl_outputs['transform'], onnx_outputs[0])
        print("benchmark", step['model'].__class__)
        print("scikit-learn")
        print(timeit("step['model'].transform(X_digits[:1])",
                     number=10000, globals=globals()))
    else:
        compare_objects(skl_outputs['predict_proba'], onnx_outputs[1])
        print("benchmark", step['model'].__class__)
        print("scikit-learn")
        print(timeit("step['model'].predict_proba(X_digits[:1])",
                     number=10000, globals=globals()))
    print("onnxruntime")
    print(timeit("sess.run(None, {'input': X_digits[:1].astype(np.float32)})",
                 number=10000, globals=globals()))

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
