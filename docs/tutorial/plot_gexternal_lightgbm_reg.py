# SPDX-License-Identifier: Apache-2.0


"""
.. _example-lightgbm-reg:

Convert a pipeline with a LightGBM regressor
============================================

.. index:: LightGBM

The discrepancies observed when using float and TreeEnsemble operator
(see :ref:`l-example-discrepencies-float-double`)
explains why the converter for *LGBMRegressor* may introduce significant
discrepancies even when it is used with float tensors.

Library *lightgbm* is implemented with double. A random forest regressor
with multiple trees computes its prediction by adding the prediction of
every tree. After being converting into ONNX, this summation becomes
:math:`\\left[\\sum\\right]_{i=1}^F float(T_i(x))`,
where *F* is the number of trees in the forest,
:math:`T_i(x)` the output of tree *i* and :math:`\\left[\\sum\\right]`
a float addition. The discrepancy can be expressed as
:math:`D(x) = |\\left[\\sum\\right]_{i=1}^F float(T_i(x)) -
\\sum_{i=1}^F T_i(x)|`.
This grows with the number of trees in the forest.

To reduce the impact, an option was added to split the node
*TreeEnsembleRegressor* into multiple ones and to do a summation
with double this time. If we assume the node if split into *a* nodes,
the discrepancies then become
:math:`D'(x) = |\\sum_{k=1}^a \\left[\\sum\\right]_{i=1}^{F/a}
float(T_{ak + i}(x)) - \\sum_{i=1}^F T_i(x)|`.

.. contents::
    :local:

Train a LGBMRegressor
+++++++++++++++++++++
"""
from distutils.version import StrictVersion
import warnings
import timeit
import numpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightgbm import LGBMRegressor
from onnxruntime import InferenceSession
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes  # noqa
from onnxmltools import __version__ as oml_version
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm  # noqa


N = 1000
X = numpy.random.randn(N, 20)
y = (numpy.random.randn(N) +
     numpy.random.randn(N) * 100 * numpy.random.randint(0, 1, 1000))

reg = LGBMRegressor(n_estimators=1000)
reg.fit(X, y)

######################################
# Register the converter for LGBMClassifier
# +++++++++++++++++++++++++++++++++++++++++
#
# The converter is implemented in :epkg:`onnxmltools`:
# `onnxmltools...LightGbm.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# lightgbm/operator_converters/LightGbm.py>`_.
# and the shape calculator:
# `onnxmltools...Regressor.py
# <https://github.com/onnx/onnxmltools/blob/master/onnxmltools/convert/
# lightgbm/shape_calculators/Regressor.py>`_.


def skl2onnx_convert_lightgbm(scope, operator, container):
    options = scope.get_options(operator.raw_operator)
    if 'split' in options:
        if StrictVersion(oml_version) < StrictVersion('1.9.2'):
            warnings.warn(
                "Option split was released in version 1.9.2 but %s is "
                "installed. It will be ignored." % oml_version)
        operator.split = options['split']
    else:
        operator.split = None
    convert_lightgbm(scope, operator, container)


update_registered_converter(
    LGBMRegressor, 'LightGbmLGBMRegressor',
    calculate_linear_regressor_output_shapes,
    skl2onnx_convert_lightgbm,
    options={'split': None})

##################################
# Convert
# +++++++
#
# We convert the same model following the two scenarios, one single
# TreeEnsembleRegressor node, or more. *split* parameter is the number of
# trees per node TreeEnsembleRegressor.

model_onnx = to_onnx(reg, X[:1].astype(numpy.float32), target_opset=14)
model_onnx_split = to_onnx(reg, X[:1].astype(numpy.float32), target_opset=14,
                           options={'split': 100})

##########################
# Discrepancies
# +++++++++++++

sess = InferenceSession(model_onnx.SerializeToString())
sess_split = InferenceSession(model_onnx_split.SerializeToString())

X32 = X.astype(numpy.float32)
expected = reg.predict(X32)
got = sess.run(None, {'X': X32})[0].ravel()
got_split = sess_split.run(None, {'X': X32})[0].ravel()

disp = numpy.abs(got - expected).sum()
disp_split = numpy.abs(got_split - expected).sum()

print("sum of discrepancies 1 node", disp)
print("sum of discrepancies split node",
      disp_split, "ratio:", disp / disp_split)

######################################
# The sum of the discrepancies were reduced 4, 5 times.
# The maximum is much better too.

disc = numpy.abs(got - expected).max()
disc_split = numpy.abs(got_split - expected).max()

print("max discrepancies 1 node", disc)
print("max discrepancies split node", disc_split, "ratio:", disc / disc_split)

################################################
# Processing time
# +++++++++++++++
#
# The processing time is slower but not much.

print("processing time no split",
      timeit.timeit(
        lambda: sess.run(None, {'X': X32})[0], number=150))
print("processing time split",
      timeit.timeit(
        lambda: sess_split.run(None, {'X': X32})[0], number=150))

#############################################
# Split influence
# +++++++++++++++
#
# Let's see how the sum of the discrepancies moves against
# the parameter *split*.

res = []
for i in tqdm(list(range(20, 170, 20)) + [200, 300, 400, 500]):
    model_onnx_split = to_onnx(reg, X[:1].astype(numpy.float32),
                               target_opset=14,
                               options={'split': i})
    sess_split = InferenceSession(model_onnx_split.SerializeToString())
    got_split = sess_split.run(None, {'X': X32})[0].ravel()
    disc_split = numpy.abs(got_split - expected).max()
    res.append(dict(split=i, disc=disc_split))

df = DataFrame(res).set_index('split')
df["baseline"] = disc
print(df)

##########################################
# Graph.

ax = df.plot(title="Sum of discrepancies against split\n"
                   "split = number of tree per node")

plt.show()
