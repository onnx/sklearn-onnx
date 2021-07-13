# SPDX-License-Identifier: Apache-2.0

"""
What is the opset number?
=========================

.. index:: opset, target opset, version

Every library is versioned. :epkg:`scikit-learn` may change
the implementation of a specific model. That happens
for example with the `SVC <https://scikit-learn.org/stable/
modules/generated/sklearn.svm.SVC.html>`_ model where
the parameter *break_ties* was added in 0.22. :epkg:`ONNX`
does also have a version called *opset number*.
Operator *ArgMin* was added in opset 1 and changed in opset
11, 12, 13. Sometimes, it is updated to extend the list
of types it supports, sometimes, it moves a parameter
into the input list. The runtime used to deploy the model
does not implement a new version, in that case, a model
must be converted by usually using the most recent opset
supported by the runtime, we call that opset the
*targeted opset*. An ONNX graph only contains
one unique opset, every node must be described following
the specifications defined by the latest opset below the
targeted opset.

This example considers an `IsolationForest
<https://scikit-learn.org/stable/modules/generated/
sklearn.ensemble.IsolationForest.html>`_ and digs into opsets.

.. contents::
    :local:

Data
++++

A simple example.
"""
from onnx.defs import onnx_opset_version
from skl2onnx import to_onnx
import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, n_features=2)

model = IsolationForest(n_estimators=3)
model.fit(X)
labels = model.predict(X)

fig, ax = plt.subplots(1, 1)
for k in (0, 1):
    ax.plot(X[labels == k, 0], X[labels == k, 1], 'o', label="cl%d" % k)
ax.set_title("Sample")

#######################################
# ONNX
# ++++


onx = to_onnx(model, X[:1].astype(numpy.float32))
print(onx)

##########################
# The last line shows the opsets.
# Let's extract it.

domains = onx.opset_import
for dom in domains:
    print("domain: %r, version: %r" % (dom.domain, dom.version))

###################################
# There are two opsets, one for standard operators,
# the other for machine learning operators.

########################################
# ONNX and opset
# ++++++++++++++
#
# The converter can convert a model to an older opset
# than the default one, from 1 to the last available one.


def get_domain_opset(onx):
    domains = onx.opset_import
    res = [{'domain': dom.domain, 'version': dom.version}
           for dom in domains]
    return {d['domain']: d['version'] for d in res}


for opset in range(1, onnx_opset_version() + 1):
    try:
        onx = to_onnx(model, X[:1].astype(numpy.float32), target_opset=opset)
    except RuntimeError as e:
        print('target: %r error: %r' % (opset, e))
        continue
    nodes = len(onx.graph.node)
    print('target: %r --> %s %d' % (opset, get_domain_opset(onx), nodes))

########################################
# It shows that the model cannot be converted for opset
# below 5. Operator `Reshape <https://github.com/onnx/
# onnx/blob/master/docs/Operators.md#Reshape>`_ changed in
# opset 5: a parameter became an input. The converter
# does not support *opset < 5* because runtimes usually do not.
#
# Other opsets
# ++++++++++++
#
# The previous example changed the opset of the main domain
# ``''`` but the other opset domain can be changed as well.

for opset in range(9, onnx_opset_version() + 1):
    for opset_ml in range(1, 3):
        tops = {'': opset, 'ai.onnx.ml': opset_ml}
        try:
            onx = to_onnx(
                model, X[:1].astype(numpy.float32), target_opset=tops)
        except RuntimeError as e:
            print('target: %r error: %r' % (opset, e))
            continue
        nodes = len(onx.graph.node)
        print('target: %r --> %s %d' % (opset, get_domain_opset(onx), nodes))
