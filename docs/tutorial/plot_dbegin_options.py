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

.. contents::
    :local:


Option *zipmap*
+++++++++++++++

Every classifier is by design converted into an ONNX graph which outputs
two results: the predicted label and the prediction probabilites
for every label. By default, the labels are integers and the
probabilites are stored in dictionaries. That's the purpose
of operator *ZipMap* added at the end of the following graph.

.. gdot::
    :script: DOT-SECTION

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = LogisticRegression()
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32))
    oinf = OnnxInference(model_def)
    print("DOT-SECTION", oinf.to_dot())

This operator is not really efficient as it copies every probabilies and
labels in a different container. This time is usually significant for
small classifiers. Then it makes sense to remove it.

.. gdot::
    :script: DOT-SECTION

    import numpy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import to_onnx
    from mlprodict.onnxrt import OnnxInference

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
    clr = LogisticRegression()
    clr.fit(X_train, y_train)

    model_def = to_onnx(clr, X_train.astype(numpy.float32),
                        options={LogisticRegression: {'zipmap': False}})
    oinf = OnnxInference(model_def)
    print("DOT-SECTION", oinf.to_dot())

There might be in the graph many classifiers, it is important to have
a way to specify which classifier should keep its *ZipMap*
and which is not. So it is possible to specify options by id.
"""

from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from pprint import pformat
from skl2onnx.common._registration import _converter_pool
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference

iris = load_iris()
X, y = iris.data, iris.target
X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
clr = LogisticRegression()
clr.fit(X_train, y_train)

model_def = to_onnx(clr, X_train.astype(numpy.float32),
                    options={id(clr): {'zipmap': False}})
oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf)

##################################
# Visually.

ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


##########################################
# We need to compare that kind of visualisation to
# what it would give with operator *ZipMap*.

model_def = to_onnx(clr, X_train.astype(numpy.float32))
oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf)

##################################
# Visually.

ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


#######################################
# Using function *id* has one flaw: it is not pickable.
# It is just better to use strings.

model_def = to_onnx(clr, X_train.astype(numpy.float32),
                    options={'zipmap': False})
oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf)


##################################
# Visually.

ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


#######################################
# Option in a pipeline
# ++++++++++++++++++++
#
# In a pipeline, :epkg:`sklearn-onnx` uses the same
# name convention.


pipe = Pipeline([
    ('norm', MinMaxScaler()),
    ('clr', LogisticRegression())
])
pipe.fit(X_train, y_train)

model_def = to_onnx(pipe, X_train.astype(numpy.float32),
                    options={'clr__zipmap': False})
oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf)

##################################
# Visually.

ax = plot_graphviz(oinf.to_dot())
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


#######################################
# Option *raw_scores*
# +++++++++++++++++++
#
# Every classifier is converted in a graph which
# returns probabilities by default. But many models
# compute unscaled *raw_scores*.
# First, with probabilities:


pipe = Pipeline([
    ('norm', MinMaxScaler()),
    ('clr', LogisticRegression())
])
pipe.fit(X_train, y_train)

model_def = to_onnx(
    pipe, X_train.astype(numpy.float32),
    options={id(pipe): {'zipmap': False}})

oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf.run({'X': X.astype(numpy.float32)[:5]}))


#######################################
# Then with raw scores:

model_def = to_onnx(
    pipe, X_train.astype(numpy.float32),
    options={id(pipe): {'raw_scores': True, 'zipmap': False}})

oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf.run({'X': X.astype(numpy.float32)[:5]}))


#########################################
# It did not seem to work... We need to tell
# that applies on a specific part of the pipeline
# and not the whole pipeline.

model_def = to_onnx(
    pipe, X_train.astype(numpy.float32),
    options={id(pipe.steps[1][1]): {'raw_scores': True, 'zipmap': False}})

oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf.run({'X': X.astype(numpy.float32)[:5]}))

###########################################
# There are negative values. That works.
# Strings are still easier to use.

model_def = to_onnx(
    pipe, X_train.astype(numpy.float32),
    options={'clr__raw_scores': True, 'clr__zipmap': False})

oinf = OnnxInference(model_def, runtime='python_compiled')
print(oinf.run({'X': X.astype(numpy.float32)[:5]}))


#########################################
# Negative figures. We still have raw scores.

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
    name = k.replace('Sklearn', '')
    print('%s%s %r' % (name, " " * (30 - len(name)), opts))
    for o in opts:
        all_opts.add(o)

print('all options:', pformat(list(sorted(all_opts))))
