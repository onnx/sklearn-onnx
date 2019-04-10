
========================
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

