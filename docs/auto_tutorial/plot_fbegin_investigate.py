# SPDX-License-Identifier: Apache-2.0

"""
Intermediate results and investigation
======================================

.. index:: investigate, intermediate results

There are many reasons why a user wants more than using
the converted model into ONNX. Intermediate results may be
needed, the output of every node in the graph. The ONNX
may need to be altered to remove some nodes.
Transfer learning is usually removing the last layers of
a deep neural network. Another reaason is debugging.
It often happens that the runtime fails to compute the predictions
due to a shape mismatch. Then it is useful the get the shape
of every intermediate result. This example looks into two
ways of doing it.

.. contents::
    :local:

Look into pipeline steps
++++++++++++++++++++++++

The first way is a tricky one: it overloads
methods *transform*, *predict* and *predict_proba*
to keep a copy of inputs and outputs. It then goes
through every step of the pipeline. If the pipeline
has *n* steps, it converts the pipeline with step 1,
then the pipeline with steps 1, 2, then 1, 2, 3...
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference
import numpy
from onnxruntime import InferenceSession
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from skl2onnx import to_onnx
from skl2onnx.helpers import collect_intermediate_steps
from skl2onnx.common.data_types import FloatTensorType

###########################
# The pipeline.

data = load_iris()
X = data.data

pipe = Pipeline(steps=[
    ('std', StandardScaler()),
    ('km', KMeans(3))
])
pipe.fit(X)

#################################
# The function goes through every step,
# overloads the methods *transform* and
# returns an ONNX graph for every step.
steps = collect_intermediate_steps(
    pipe, "pipeline",
    [("X", FloatTensorType([None, X.shape[1]]))])

#####################################
# We call method transform to population the
# cache the overloaded methods *transform* keeps.
pipe.transform(X)

#######################################
# We compute every step and compare
# ONNX and scikit-learn outputs.

for step in steps:
    print('----------------------------')
    print(step['model'])
    onnx_step = step['onnx_step']
    sess = InferenceSession(onnx_step.SerializeToString())
    onnx_outputs = sess.run(None, {'X': X.astype(numpy.float32)})
    onnx_output = onnx_outputs[-1]
    skl_outputs = step['model']._debug.outputs['transform']

    # comparison
    diff = numpy.abs(skl_outputs.ravel() - onnx_output.ravel()).max()
    print("difference", diff)

# That was the first way: dynamically overwrite
# every method transform or predict in a scikit-learn
# pipeline to capture the input and output of every step,
# compare them to the output produced by truncated ONNX
# graphs built from the first one.
#
#####################################
# Python runtime to look into every node
# ++++++++++++++++++++++++++++++++++++++
#
# The python runtime may be useful to easily look
# into every node of the ONNX graph.
# This option can be used to check when the computation
# fails due to nan values or a dimension mismatch.


onx = to_onnx(pipe, X[:1].astype(numpy.float32))

oinf = OnnxInference(onx)
oinf.run({'X': X[:2].astype(numpy.float32)},
         verbose=1, fLOG=print)

###################################
# And to get a sense of the intermediate results.

oinf.run({'X': X[:2].astype(numpy.float32)},
         verbose=3, fLOG=print)

# This way is usually better if you need to investigate
# issues within the code of the runtime for an operator.
#
#################################
# Final graph
# +++++++++++

ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
