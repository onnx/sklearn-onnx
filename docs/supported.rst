
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

.. toctree::
    :maxdepth: 1

    supported_covered
    supported_impl
    supported_pipe
    supported_onnx
    graphs/index
