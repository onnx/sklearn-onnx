# SPDX-License-Identifier: Apache-2.0


"""
.. _l-benchmark-cdist:

Compare CDist with scipy
========================

The following example focuses on one particular operator,
CDist and compares its execution time between
*onnxruntime* and *scipy*.

.. contents::
    :local:

ONNX Graph with CDist
+++++++++++++++++++++

`cdist <https://docs.scipy.org/doc/scipy/reference/
generated/scipy.spatial.distance.cdist.html>`_
function computes pairwise distances.
"""
from pprint import pprint
from timeit import Timer
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pandas import DataFrame
import onnx
import onnxruntime as rt
from onnxruntime import InferenceSession
import skl2onnx
from skl2onnx.algebra.custom_ops import OnnxCDist
from skl2onnx.common.data_types import FloatTensorType

X = np.ones((2, 4), dtype=np.float32)
Y = np.ones((3, 4), dtype=np.float32)
Y *= 2
print(cdist(X, Y, metric='euclidean'))

####################################
# ONNX

op = OnnxCDist('X', 'Y', op_version=12, output_names=['Z'],
               metric='euclidean')
onx = op.to_onnx({'X': X, 'Y': Y},
                 outputs=[('Z', FloatTensorType())])
print(onx)


########################################
# CDist and onnxruntime
# +++++++++++++++++++++
#
# We compute the output of CDist operator
# with onnxruntime.

sess = InferenceSession(onx.SerializeToString())
res = sess.run(None, {'X': X, 'Y': Y})
print(res)

#####################################
# Benchmark
# +++++++++
#
# Let's compare onnxruntime and scipy.


def measure_time(name, stmt, context, repeat=100, number=20):
    tim = Timer(stmt, globals=context)
    res = np.array(
        tim.repeat(repeat=repeat, number=number))
    res /= number
    mean = np.mean(res)
    dev = np.mean(res ** 2)
    dev = (dev - mean**2) ** 0.5
    return dict(
        average=mean, deviation=dev, min_exec=np.min(res),
        max_exec=np.max(res), repeat=repeat, number=number,
        nrows=context['X'].shape[0], ncols=context['Y'].shape[1],
        name=name)


##############################
# scipy

time_scipy = measure_time(
    "scipy", "cdist(X, Y)",
    context={'cdist': cdist, 'X': X, 'Y': Y})
pprint(time_scipy)


###############################
# onnxruntime

time_ort = measure_time(
    "ort", "sess.run(None, {'X': X, 'Y': Y})",
    context={'sess': sess, 'X': X, 'Y': Y})
pprint(time_ort)

############################################
# Longer benchmark

metrics = []
for dim in tqdm([10, 100, 1000, 10000]):
    # We cannot change the number of column otherwise
    # we need to create a new graph.
    X = np.random.randn(dim, 4).astype(np.float32)
    Y = np.random.randn(10, 4).astype(np.float32)

    time_scipy = measure_time(
        "scipy", "cdist(X, Y)",
        context={'cdist': cdist, 'X': X, 'Y': Y})
    time_ort = measure_time(
        "ort", "sess.run(None, {'X': X, 'Y': Y})",
        context={'sess': sess, 'X': X, 'Y': Y})
    metric = dict(N=dim, scipy=time_scipy['average'],
                  ort=time_ort['average'])
    metrics.append(metric)

df = DataFrame(metrics)
df['scipy/ort'] = df['scipy'] / df['ort']
print(df)

df.plot(x='N', y=['scipy/ort'])

#################################
# **Versions used for this example**

print("numpy:", np.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
