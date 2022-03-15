# SPDX-License-Identifier: Apache-2.0

"""
Benchmark ONNX conversion
=========================

.. index:: benchmark

Example :ref:`l-simple-deploy-1` converts a simple model.
This example takes a similar example but on random data
and compares the processing time required by each option
to compute predictions.

.. contents::
    :local:


Training a pipeline
+++++++++++++++++++
"""
import numpy
from pandas import DataFrame
from tqdm import tqdm
from sklearn import config_context
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    VotingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlprodict.onnxrt import OnnxInference
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from skl2onnx.tutorial import measure_time


N = 11000
X, y = make_regression(N, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.01)
print("Train shape", X_train.shape)
print("Test shape", X_test.shape)

reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg.fit(X_train, y_train)

#################################
# Measure the processing time
# +++++++++++++++++++++++++++
#
# We use function :func:`skl2onnx.tutorial.measure_time`.
# The page about `assume_finite <https://scikit-learn.org/
# stable/modules/generated/sklearn.config_context.html>`_
# may be useful if you need to optimize the prediction.
# We measure the processing time per observation whether
# or not an observation belongs to a batch or is a single one.

sizes = [(1, 50), (10, 50), (1000, 10), (10000, 5)]

with config_context(assume_finite=True):
    obs = []
    for batch_size, repeat in tqdm(sizes):
        context = {"ereg": ereg, 'X': X_test[:batch_size]}
        mt = measure_time(
            "ereg.predict(X)", context, div_by_number=True,
            number=10, repeat=repeat)
        mt['size'] = context['X'].shape[0]
        mt['mean_obs'] = mt['average'] / mt['size']
        obs.append(mt)

df_skl = DataFrame(obs)
df_skl

#####################################
# Graphe.

df_skl.set_index('size')[['mean_obs']].plot(
    title="scikit-learn", logx=True, logy=True)

###############################
# ONNX runtime
# ++++++++++++
#
# The same is done with the two ONNX runtime
# available.

onx = to_onnx(ereg, X_train[:1].astype(numpy.float32),
              target_opset=14)
sess = InferenceSession(onx.SerializeToString())
oinf = OnnxInference(onx, runtime="python_compiled")

obs = []
for batch_size, repeat in tqdm(sizes):

    # scikit-learn
    context = {"ereg": ereg, 'X': X_test[:batch_size].astype(numpy.float32)}
    mt = measure_time(
        "ereg.predict(X)", context, div_by_number=True,
        number=10, repeat=repeat)
    mt['size'] = context['X'].shape[0]
    mt['skl'] = mt['average'] / mt['size']

    # onnxruntime
    context = {"sess": sess, 'X': X_test[:batch_size].astype(numpy.float32)}
    mt2 = measure_time(
        "sess.run(None, {'X': X})[0]", context, div_by_number=True,
        number=10, repeat=repeat)
    mt['ort'] = mt2['average'] / mt['size']

    # mlprodict
    context = {"oinf": oinf, 'X': X_test[:batch_size].astype(numpy.float32)}
    mt2 = measure_time(
        "oinf.run({'X': X})['variable']", context, div_by_number=True,
        number=10, repeat=repeat)
    mt['pyrt'] = mt2['average'] / mt['size']

    # end
    obs.append(mt)


df = DataFrame(obs)
df

#####################################
# Graph.

df.set_index('size')[['skl', 'ort', 'pyrt']].plot(
    title="Average prediction time per runtime",
    logx=True, logy=True)

#####################################
# :epkg:`ONNX` runtimes are much faster than :epkg:`scikit-learn`
# to predict one observation. :epkg:`scikit-learn` is optimized
# for training, for batch prediction. That explains why
# :epkg:`scikit-learn` and ONNX runtimes seem to converge
# for big batches. They use similar implementation,
# parallelization and languages (:epkg:`C++`, :epkg:`openmp`).
