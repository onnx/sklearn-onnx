..  SPDX-License-Identifier: Apache-2.0


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

.. _l-converter-list:

Covered Converters
==================

.. covered-sklearn-ops::

Converters Documentation
========================

.. supported-sklearn-ops::

Pipeline
========

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
<https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_
and `ONNX-ML Operators
<https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md>`_.
Associated to `onnxruntime <https://github.com/Microsoft/onnxruntime>`_,
the mapping makes it easier to easily check the output
of the *ONNX* operators on any data as shown
in example :ref:`l-onnx-operators`.

.. supported-onnx-ops::
