
.. _l-conv-options:

=======================
Converters with options
=======================

Most of the converters always produce the same converted model
which computes the same outputs as the original model.
However, some of them do not and the user may need to alter the
conversion by giving additional information to the converter.
Below is the list of converters which enable this mechanism.

.. autofunction:: skl2onnx.operator_converters.TextVectorizer.convert_sklearn_text_vectorizer
