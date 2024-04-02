..  SPDX-License-Identifier: Apache-2.0


.. _l-conv-options:

=======================
Converters with options
=======================

Most of the converters always produce the same converted model
which computes the same outputs as the original model.
However, some of them do not and the user may need to alter the
conversion by giving additional information to the converter through
functions :func:`convert_sklearn <skl2onnx.convert_sklearn>`
or :func:`to_onnx <skl2onnx.to_onnx>`.
Every option ends up creating a different ONNX graph.
Below is the list of models which enable this mechanism.

GaussianProcessRegressor, NearestNeighbors
==========================================

.. index:: pairwise distances, cdist

Both models require to compure pairwise distances.
Function :func:`onnx_cdist <skl2onnx.algebra.complex_functions.onnx_cdist>`
produces this part of the graph but there exist two options.
The first one is using *Scan* operator, the second one is
using a dedicated operator called *CDist* which is not part
of the regular ONNX operator until issue
`#2442 <https://github.com/onnx/onnx/issues/2442>`_
is addressed. By default, *Scan* is used, *CDist* can be used
by giving:

::

    options={GaussianProcessRegressor: {'optim': 'cdist'}}

Previous line enables the optimization for every
model *GaussianProcessRegressor* but it can be done
only for one model by using:

::

    options={id(model): {'optim': 'cdist'}}

TfidfVectorizer, CountVectorizer
================================

.. autofunction:: skl2onnx.operator_converters.text_vectoriser.convert_sklearn_text_vectorizer

Classifiers
===========

Converters for classifiers implement multiple options.

ZipMap
------

The operator *ZipMap* produces a list of dictionaries.
It repeats class names or ids but that's not necessary
(see issue `#2149 <https://github.com/onnx/onnx/issues/2149>`_).
By default, ZipMap operator is added, it can be deactivated by using:

::

    options={type(model): {'zipmap': False}}

It is implemented by PR `327 <https://github.com/onnx/sklearn-onnx/pull/327>`_.

Class information
-----------------

Class information is usually repeated in the ONNX operator
which classifies and the output of the ZipMap operator
(see issue `2149 <https://github.com/onnx/onnx/issues/2149>`_).
The following option can remove string information and class ids
in the ONNX operator to get smaller models.

::

    options={type(model): {'nocl': True}}

Classes are replaced by integers from 0 to the number of classes.

Raw scores
----------

Almost all classifiers are converted in order to get probabilities
and not raw scores. That's the default behaviour. It can be deactivated
by using option:

::

    options={type(model): {'raw_scores': True}}

It is implemented by PR `308 <https://github.com/onnx/sklearn-onnx/pull/308>`_.

Pickability and Pipeline
========================

The proposed way to specify options is not always pickable.
Function ``id(model)`` depends on the execution and map an option
to one class may be not enough to customize the conversion.
However, it is possible to specify an option the same way
parameters are referenced in a *scikit-learn* pipeline
with method `get_params <https://scikit-learn.org/stable/modules/generated/
sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.get_params>`_.
Following syntax are supported:

::

    pipe = Pipeline([('pca', PCA()), ('classifier', LogisticRegression())])

    options = {'classifier': {'zipmap': False}}

Or

::

    options = {'classifier__zipmap': False}

Options applied to one model, not a pipeline as the converter
replaces the pipeline structure by a single onnx graph.
Following that rule, option *zipmap* would not have any impact
if applied to a pipeline and to the last step of the pipeline.
However, because there is no ambiguity about what the conversion
should be, for options *zipmap* and *nocl*, the following
options would have the same effect:

::

    pipe = Pipeline([('pca', PCA()), ('classifier', LogisticRegression())])

    options = {id(pipe.steps[-1][1]): {'zipmap': False}}
    options = {id(pipe): {'zipmap': False}}
    options = {'classifier': {'zipmap': False}}
    options = {'classifier__zipmap': False}
