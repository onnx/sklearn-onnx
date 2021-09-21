# SPDX-License-Identifier: Apache-2.0

"""
.. _l-example-discrepencies-float-double:

Issues when switching to float
==============================

.. index:: float, double, discrepencies

Most models in :epkg:`scikit-learn` do computation with double,
not float. Most models in deep learning use float because
that's the most common situation with GPU. ONNX was initially
created to facilitate the deployment of deep learning models
and that explains why many converters assume the converted models
should use float. That assumption does not usually harm
the predictions, the conversion to float introduce small
discrepencies compare to double predictions.
That assumption is usually true if the prediction
function is continuous, :math:`y = f(x)`, then
:math:`dy = f'(x) dx`. We can determine an upper bound
to the discrepencies :
:math:`\\Delta(y) \\leqslant \\sup_x \\left\\Vert f'(x)\\right\\Vert dx`.
*dx* is the discrepency introduced by a float conversion,
``dx = x - numpy.float32(x)``.

However, that's not the case for every model. A decision tree
trained for a regression is not a continuous function. Therefore,
even a small *dx* may introduce a huge discrepency. Let's look into
an example which always produces discrepencies and some ways
to overcome this situation.

.. contents::
    :local:

More into the issue
+++++++++++++++++++

The below example is built to fail.
It contains integer features with different order
of magnitude rounded to integer. A decision tree compares
features to thresholds. In most cases, float and double
comparison gives the same result. We denote
:math:`[x]_{f32}` the conversion (or cast)
``numpy.float32(x)``.

.. math::

    x \\leqslant y = [x]_{f32} \\leqslant [y]_{f32}

However, the probability that both comparisons give
different results is not null. The following graph shows
the discord areas.
"""
from mlprodict.sklapi import OnnxPipeline
from skl2onnx.sklapi import CastTransformer, CastRegressor
from skl2onnx import to_onnx
from mlprodict.onnx_conv import to_onnx as to_onnx_extended
from mlprodict.onnxrt import OnnxInference
from onnxruntime import InferenceSession
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
import numpy
import matplotlib.pyplot as plt


def area_mismatch_rule(N, delta, factor, rule=None):
    if rule is None:
        def rule(t): return numpy.float32(t)
    xst = []
    yst = []
    xsf = []
    ysf = []
    for x in range(-N, N):
        for y in range(-N, N):
            dx = (1. + x * delta) * factor
            dy = (1. + y * delta) * factor
            c1 = 1 if numpy.float64(dx) <= numpy.float64(dy) else 0
            c2 = 1 if numpy.float32(dx) <= rule(dy) else 0
            key = abs(c1 - c2)
            if key == 1:
                xsf.append(dx)
                ysf.append(dy)
            else:
                xst.append(dx)
                yst.append(dy)
    return xst, yst, xsf, ysf


delta = 36e-10
factor = 1
xst, yst, xsf, ysf = area_mismatch_rule(100, delta, factor)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(xst, yst, '.', label="agree")
ax.plot(xsf, ysf, '.', label="disagree")
ax.set_title("Region where x <= y and (float)x <= (float)y agree")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot([min(xst), max(xst)], [min(yst), max(yst)], 'k--')
ax.legend()


#####################################
# The pipeline and the data
# +++++++++++++++++++++++++
#
# We can now build an example where the learned decision tree
# does many comparisons in this discord area. This is done
# by rounding features to integers, a frequent case
# happening when dealing with categorical features.


X, y = make_regression(10000, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y)

Xi_train, yi_train = X_train.copy(), y_train.copy()
Xi_test, yi_test = X_test.copy(), y_test.copy()
for i in range(X.shape[1]):
    Xi_train[:, i] = (Xi_train[:, i] * 2 ** i).astype(numpy.int64)
    Xi_test[:, i] = (Xi_test[:, i] * 2 ** i).astype(numpy.int64)

max_depth = 10

model = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model.fit(Xi_train, yi_train)

#######################################
# The discrepencies
# +++++++++++++++++
#
# Let's reuse the function implemented in the
# first example :ref:`l-diff-dicrepencies` and
# look into the conversion.


def diff(p1, p2):
    p1 = p1.ravel()
    p2 = p2.ravel()
    d = numpy.abs(p2 - p1)
    return d.max(), (d / numpy.abs(p1)).max()


onx = to_onnx(model, Xi_train[:1].astype(numpy.float32))

sess = InferenceSession(onx.SerializeToString())

X32 = Xi_test.astype(numpy.float32)

skl = model.predict(X32)
ort = sess.run(None, {'X': X32})[0]

print(diff(skl, ort))

###################################
# The discrepencies are significant.
# The ONNX model keeps float at every step.
#
# .. blockdiag::
#
#    diagram {
#      x_float32 -> normalizer -> y_float32 -> dtree -> z_float32
#    }
#
# In :epkg:`scikit-learn`:
#
# .. blockdiag::
#
#    diagram {
#      x_float32 -> normalizer -> y_double -> dtree -> z_double
#    }
#
# CastTransformer
# +++++++++++++++
#
# We could try to use double everywhere. Unfortunately,
# :epkg:`ONNX ML Operators` only allows float coefficients
# for the operator *TreeEnsembleRegressor*. We may want
# to compromise by casting the output of the normalizer into
# float in the :epkg:`scikit-learn` pipeline.
#
# .. blockdiag::
#
#    diagram {
#      x_float32 -> normalizer -> y_double ->
#      cast -> y_float -> dtree -> z_float
#    }
#


model2 = Pipeline([
    ('scaler', StandardScaler()),
    ('cast', CastTransformer()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model2.fit(Xi_train, yi_train)

##########################################
# The discrepencies.

onx2 = to_onnx(model2, Xi_train[:1].astype(numpy.float32))

sess2 = InferenceSession(onx2.SerializeToString())

skl2 = model2.predict(X32)
ort2 = sess2.run(None, {'X': X32})[0]

print(diff(skl2, ort2))

######################################
# That still fails because the normalizer
# in :epkg:`scikit-learn` and in :epkg:`ONNX`
# use different types. The cast still happens and
# the *dx* is still here. To remove it, we need to use
# double in ONNX normalizer.

model3 = Pipeline([
    ('cast64', CastTransformer(dtype=numpy.float64)),
    ('scaler', StandardScaler()),
    ('cast', CastTransformer()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model3.fit(Xi_train, yi_train)
onx3 = to_onnx(model3, Xi_train[:1].astype(numpy.float32),
               options={StandardScaler: {'div': 'div_cast'}})

sess3 = InferenceSession(onx3.SerializeToString())

skl3 = model3.predict(X32)
ort3 = sess3.run(None, {'X': X32})[0]

print(diff(skl3, ort3))

#################################
# It works. That also means that it is difficult to change
# the computation type when a pipeline includes a discontinuous
# function. It is better to keep the same types all along
# before using a decision tree.
#
# Sledgehammer
# ++++++++++++
#
# The idea here is to always train the next step based
# on ONNX outputs. That way, every step of the pipeline
# is trained based on ONNX output.
#
# * Trains the first step.
# * Converts the step into ONNX
# * Computes ONNX outputs.
# * Trains the second step on these outputs.
# * Converts the second step into ONNX.
# * Merges it with the first step.
# * Computes ONNX outputs of the merged two first steps.
# * ...
#
# It is implemented in
# class :epkg:`OnnxPipeline`.


model_onx = OnnxPipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeRegressor(max_depth=max_depth))
])

model_onx.fit(Xi_train, yi_train)

#############################################
# The conversion.

try:
    onx4 = to_onnx(model_onx, Xi_train[:1].astype(numpy.float32))
except ValueError as e:
    print("Failing due to %r.\nYou need to update mlprodict." % e)
    import sys
    sys.exit(0)

sess4 = InferenceSession(onx4.SerializeToString())

skl4 = model_onx.predict(X32)
ort4 = sess4.run(None, {'X': X32})[0]

print(diff(skl4, ort4))

#################################
# It works too in a more simple way.

########################################
# No discrepencies at all?
# ++++++++++++++++++++++++
#
# Is it possible to get no error at all?
# There is one major obstacle: :epkg:`scikit-learn`
# stores the predicted values in every leave with double
# (`_tree.pyx - _get_value_ndarray
# <https://github.com/scikit-learn/scikit-learn/blob/master/
# sklearn/tree/_tree.pyx#L1096>`_), :epkg:`ONNX` defines the
# the predicted values as floats: :epkg:`TreeEnsembleRegressor`.
# What can we do to solve it?
# What if we could extend ONNX specifications to support
# double instead of floats.
# We reuse what was developped in example
# `Other way to convert <http://www.xavierdupre.fr/app/
# mlprodict/helpsphinx/notebooks/onnx_discrepencies.html
# ?highlight=treeensembleregressordouble#other-way-to-convert>`_
# and a custom ONNX node `TreeEnsembleRegressorDouble
# <http://www.xavierdupre.fr/app/mlprodict/helpsphinx/api/onnxrt_ops.html
# ?highlight=treeensembleregressordouble#treeensembleregressordouble>`_.


tree = DecisionTreeRegressor(max_depth=max_depth)
tree.fit(Xi_train, yi_train)

model_onx = to_onnx_extended(tree, Xi_train[:1].astype(numpy.float64),
                             rewrite_ops=True)

oinf5 = OnnxInference(model_onx, runtime='python_compiled')
print(oinf5)

##########################################
# Let's measure the discrepencies.

X64 = Xi_test.astype(numpy.float64)
skl5 = tree.predict(X64)
ort5 = oinf5.run({'X': X64})['variable']

############################################
# Perfect, no discrepencies at all.

print(diff(skl5, ort5))

##############################################
# CastRegressor
# +++++++++++++
#
# The previous example demonstrated the type difference for
# the predicted values explains the small differences between
# :epkg:`scikit-learn` and :epkg:`onnxruntime`. But it does not
# with the current ONNX. Another option is to cast the
# the predictions into floats in the :epkg:`scikit-learn` pipeline.


ctree = CastRegressor(DecisionTreeRegressor(max_depth=max_depth))
ctree.fit(Xi_train, yi_train)

onx6 = to_onnx(ctree, Xi_train[:1].astype(numpy.float32))

sess6 = InferenceSession(onx6.SerializeToString())

skl6 = ctree.predict(X32)
ort6 = sess6.run(None, {'X': X32})[0]

print(diff(skl6, ort6))

##############################
# Success!
