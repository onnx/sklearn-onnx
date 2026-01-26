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

Following sections show how to create a custom converter.
It assumes this new converter is not meant to be added to
this package but only to be registered and used when converting
a pipeline. To to contribute and add a converter
for a :epkg:`scikit-learn` model, the logic is still the same,
only the converter registration changes. `PR 737
<https://github.com/onnx/sklearn-onnx/pull/737>`_ can be used as
an example.

.. toctree::
    :maxdepth: 1

    auto_tutorial/plot_icustom_converter
    auto_tutorial/plot_jcustom_syntax
    auto_tutorial/plot_jfunction_transformer
    auto_tutorial/plot_k1_custom_converter_onnxscript
    auto_tutorial/plot_kcustom_converter_wrapper
    auto_tutorial/plot_lcustom_options
    auto_tutorial/plot_mcustom_parser
