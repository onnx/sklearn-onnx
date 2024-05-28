# SPDX-License-Identifier: Apache-2.0

"""
One model, many possible conversions with options
=================================================

.. index:: options

There is not one way to convert a model. A new operator
might have been added in a newer version of :epkg:`ONNX`
and that speeds up the converted model. The rational choice
would be to use this new operator but what means the associated
runtime has an implementation for it. What if two different
users needs two different conversion for the same model?
Let's see how this may be done.

Option *zipmap*
+++++++++++++++

Every classifier is by design converted into an ONNX graph which outputs
two results: the predicted label and the prediction probabilites
for every label. By default, the labels are integers and the
probabilites are stored in dictionaries. That's the purpose
of operator *ZipMap* added at the end of the following graph.

.. runpython::
    import numpy
    from onnx.helper import printable_graph
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import to_onnx

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = LogisticRegression(max_iter=1000)
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32))
    print(printable_graph(model_def.graph))

This operator is not really efficient as it copies every probabilies and
labels in a different container. This time is usually significant for
small classifiers. Then it makes sense to remove it.

.. runpython::

    import numpy
    from onnx.helper import printable_graph
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import to_onnx

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = LogisticRegression(max_iter=1000)
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        options={LogisticRegression: {'zipmap': False}})
    print(printable_graph(model_def.graph))

There might be in the graph many classifiers, it is important to have
a way to specify which classifier should keep its *ZipMap*
and which is not. So it is possible to specify options by id.
"""

from pprint import pformat
import numpy
from onnx.reference import ReferenceEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx.common._registration import _converter_pool
from skl2onnx import to_onnx
from onnxruntime import InferenceSession

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
clr = LogisticRegression()
clr.fit(X_train, y_train)

model_def = to_onnx(
    clr, X_train.astype(numpy.float32), options={id(clr): {"zipmap": False}}
)
oinf = ReferenceEvaluator(model_def)
print(oinf)


#######################################
# Using function *id* has one flaw: it is not pickable.
# It is just better to use strings.

model_def = to_onnx(clr, X_train.astype(numpy.float32), options={"zipmap": False})
oinf = ReferenceEvaluator(model_def)
print(oinf)


#######################################
# Option in a pipeline
# ++++++++++++++++++++
#
# In a pipeline, :epkg:`sklearn-onnx` uses the same
# name convention.


pipe = Pipeline([("norm", MinMaxScaler()), ("clr", LogisticRegression())])
pipe.fit(X_train, y_train)

model_def = to_onnx(pipe, X_train.astype(numpy.float32), options={"clr__zipmap": False})
oinf = ReferenceEvaluator(model_def)
print(oinf)

#######################################
# Option *raw_scores*
# +++++++++++++++++++
#
# Every classifier is converted in a graph which
# returns probabilities by default. But many models
# compute unscaled *raw_scores*.
# First, with probabilities:


pipe = Pipeline([("norm", MinMaxScaler()), ("clr", LogisticRegression())])
pipe.fit(X_train, y_train)

model_def = to_onnx(
    pipe, X_train.astype(numpy.float32), options={id(pipe): {"zipmap": False}}
)

oinf = ReferenceEvaluator(model_def)
print(oinf.run(None, {"X": X.astype(numpy.float32)[:5]}))


#######################################
# Then with raw scores:

model_def = to_onnx(
    pipe,
    X_train.astype(numpy.float32),
    options={id(pipe): {"raw_scores": True, "zipmap": False}},
)

oinf = ReferenceEvaluator(model_def)
print(oinf.run(None, {"X": X.astype(numpy.float32)[:5]}))

#########################################
# It did not seem to work... We need to tell
# that applies on a specific part of the pipeline
# and not the whole pipeline.

model_def = to_onnx(
    pipe,
    X_train.astype(numpy.float32),
    options={id(pipe.steps[1][1]): {"raw_scores": True, "zipmap": False}},
)

oinf = ReferenceEvaluator(model_def)
print(oinf.run(None, {"X": X.astype(numpy.float32)[:5]}))

###########################################
# There are negative values. That works.
# Strings are still easier to use.

model_def = to_onnx(
    pipe,
    X_train.astype(numpy.float32),
    options={"clr__raw_scores": True, "clr__zipmap": False},
)

oinf = ReferenceEvaluator(model_def)
print(oinf.run(None, {"X": X.astype(numpy.float32)[:5]}))


#########################################
# Negative figures. We still have raw scores.

#######################################
# Option *decision_path*
# ++++++++++++++++++++++
#
# *scikit-learn* implements a function to retrieve the
# decision path. It can be enabled by option *decision_path*.

clrrf = RandomForestClassifier(n_estimators=2, max_depth=2)
clrrf.fit(X_train, y_train)
clrrf.predict(X_test[:2])
paths, n_nodes_ptr = clrrf.decision_path(X_test[:2])
print(paths.todense())

model_def = to_onnx(
    clrrf,
    X_train.astype(numpy.float32),
    options={id(clrrf): {"decision_path": True, "zipmap": False}},
)
sess = InferenceSession(
    model_def.SerializeToString(), providers=["CPUExecutionProvider"]
)

##########################################
# The model produces 3 outputs.

print([o.name for o in sess.get_outputs()])

##########################################
# Let's display the last one.

res = sess.run(None, {"X": X_test[:2].astype(numpy.float32)})
print(res[-1])

############################################################
# List of available options
# +++++++++++++++++++++++++
#
# Options are registered for every converted to detect any
# supported options while running the conversion.


all_opts = set()
for k, v in sorted(_converter_pool.items()):
    opts = v.get_allowed_options()
    if not isinstance(opts, dict):
        continue
    name = k.replace("Sklearn", "")
    print("%s%s %r" % (name, " " * (30 - len(name)), opts))
    for o in opts:
        all_opts.add(o)

print("all options:", pformat(list(sorted(all_opts))))
