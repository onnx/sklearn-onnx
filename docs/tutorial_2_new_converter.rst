..  SPDX-License-Identifier: Apache-2.0

A custom converter for a custom model
=====================================

When :epkg:`sklearn-onnx` converts a :epkg:`scikit-learn`
pipeline, it looks into every transformer and predictor
and fetches the associated converter. The resulting
ONNX graph combines the outcome of every converter
in a single graph. If a model does not have its converter,
it displays an error message telling it misses a converter.

.. runpython::
    :showcode:

    import numpy
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import to_onnx


    class MyLogisticRegression(LogisticRegression):
        pass


    X = numpy.array([[0, 0.1]])
    try:
        to_onnx(MyLogisticRegression(), X)
    except Exception as e:
        print(e)

Following section shows how to create a custom converter.

.. toctree::
    :maxdepth: 1

    auto_tutorial/plot_icustom_converter
    auto_tutorial/plot_jcustom_syntax
    auto_tutorial/plot_kcustom_converter_wrapper
    auto_tutorial/plot_lcustom_options
    auto_tutorial/plot_mcustom_parser
