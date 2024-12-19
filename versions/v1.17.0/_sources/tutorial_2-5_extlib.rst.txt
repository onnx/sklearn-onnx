..  SPDX-License-Identifier: Apache-2.0

Write converters for other libraries
====================================

*sklearn-onnx* only converts models from *scikit-learn*. It
implements a mechanism to register converters from other libraries.
Converters for models from other libraries will not be added to
*sklearn-onnx*. Every library has its own maintenance cycle and
it would become difficult to maintain a package having too many
dependencies. Following examples were added to show how to 
develop converters for new libraries.

.. toctree::
    :maxdepth: 1

    auto_tutorial/plot_wext_pyod_forest
