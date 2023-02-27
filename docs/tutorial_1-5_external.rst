..  SPDX-License-Identifier: Apache-2.0

Using converters from other libraries
=====================================

Before starting writing our own converter,
we can use some available in other libraries
than :epkg:`sklearn-onnx`. :epkg:`onnxmltools` implements
converters for :epkg:`xgboost` and :epkg:`LightGBM`.
Following examples show how to use the conveter when the
model are part of a pipeline.

.. toctree::
    :maxdepth: 1

    auto_tutorial/plot_gexternal_lightgbm
    auto_tutorial/plot_gexternal_lightgbm_reg
    auto_tutorial/plot_gexternal_xgboost
    auto_tutorial/plot_gexternal_catboost
