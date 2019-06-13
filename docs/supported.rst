
=============================
Supported scikit-learn Models
=============================

*skl2onnx* currently can convert the following list
of models for *skl2onnx* :skl2onnxversion:`v`. They 
were tested using *onnxruntime* :skl2onnxversion:`rt`.
All the following classes overloads the following methods
such as :class:`OnnxSklearnPipeline` does. They wrap existing
*scikit-learn* classes by dynamically creating a new one
which inherits from :class:`OnnxOperatorMixin` which
implements *to_onnx* methods.

.. contents::
    :local:

.. _l-converter-list:

Covered and Uncovered Converters
================================

.. covered-sklearn-ops::

Documentation of Implemented Converters
=======================================

.. supported-sklearn-ops::

Supported Pipeline Operators
============================

.. autoclass:: skl2onnx.algebra.sklearn_ops.OnnxSklearnPipeline
    :members: to_onnx, to_onnx_operator, onnx_parser, onnx_shape_calculator, onnx_converter

.. autoclass:: skl2onnx.algebra.sklearn_ops.OnnxSklearnColumnTransformer
    :members: to_onnx, to_onnx_operator, onnx_parser, onnx_shape_calculator, onnx_converter

.. autoclass:: skl2onnx.algebra.sklearn_ops.OnnxSklearnFeatureUnion
    :members: to_onnx, to_onnx_operator, onnx_parser, onnx_shape_calculator, onnx_converter

Available ONNX operators
========================

*skl2onnx* maps every ONNX operators into a class
easy to insert into a graph. These operators get
dynamically added and the list depends on the installed
*ONNX* package. The documentation for these operators
can be found on github: `ONNX Operators.md
<https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
and `ONNX-ML Operators
<https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md>`_.
Associated to `onnxruntime <https://github.com/Microsoft/onnxruntime>`_,
the mapping makes it easier to easily check the output
of the *ONNX* operators on any data as shown
in example :ref:`l-onnx-operators`.

.. supported-onnx-ops::

Availability of Converters for each Opset
=========================================

Some ONNX operators converters are using were not all 
available in older version of ONNX. This version is called
*opset number*. ONNX 1.4.0 is opset 9, ONNX 1.5.0 is opset 10...
Next table shows which operator is available in which opset.
An empty cell means it is not available. Other cells
contains concatenated flags whose meaning is the following:

* ``ERROR`` means the automated process failed to give
  a appropriate status or the runtime produces predictions
  too far from the original predictions,
* ``OK``: the converter works fine and the runtime produces
  predictions almost equal to the orignal predictions,
  absolute difference is below :math:`1e-5`,
* ``e<0.01``: the converter works fine and the runtime produces
  predictions close to the orignal predictions,
  absolute difference is below :math:`0.01`,
* ``e<0.1``: the converter works fine but the runtime produces
  predictions close to the orignal predictions,
  absolute difference is below :math:`0.1`,
* ``i|j``: the model was converted for a specific opset but
  the converted ONNX is compatible with smaller opset,
  *i* is the smallest compatible opset for the main domain,
  *j* is the smallest compatible opset for the ai domain,
* ``NOBATCH``: the runtime is unable to compute the predictions
  for multiple observations at the same time, it needs to be
  called for each observation.

The model are tested through simple problems using the Iris dataset.
The datasets is split into train test datasets.

* *bin-class*: binary classification,
* *multi-class*: multi-class classification,
* *regression*: regression,
* *num-transform*: no label, only numerical features

.. supported-onnx-ops-opset::
