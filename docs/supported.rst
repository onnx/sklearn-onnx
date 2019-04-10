
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

Pipeline
========

.. autoclass:: skl2onnx.algebra.sklearn_ops.OnnxSklearnPipeline
    :members: to_onnx, to_onnx_operator, onnx_parser, onnx_shape_calculator, onnx_converter

.. autoclass:: skl2onnx.algebra.sklearn_ops.OnnxSklearnColumnTransformer
    :members: to_onnx, to_onnx_operator, onnx_parser, onnx_shape_calculator, onnx_converter

.. autoclass:: skl2onnx.algebra.sklearn_ops.OnnxSklearnFeatureUnion
    :members: to_onnx, to_onnx_operator, onnx_parser, onnx_shape_calculator, onnx_converter

Implemented Converters
======================

.. supported-sklearn-ops::

