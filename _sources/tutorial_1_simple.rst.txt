..  SPDX-License-Identifier: Apache-2.0


The easy case
=============

The easy case is when the machine learned model
can be converter into ONNX with a converting library
without writing nay specific code. That means that a converter
exists for the model or each piece of the model,
the converter produces an ONNX graph where every node
is part of the existing ONNX specifications, the runtime
used to compute the predictions implements every node
used in the ONNX graph.

.. toctree::
    :maxdepth: 1

    auto_tutorial/plot_abegin_convert_pipeline
    auto_tutorial/plot_bbegin_measure_time
    auto_tutorial/plot_cbegin_opset
    auto_tutorial/plot_dbegin_options
    auto_tutorial/plot_dbegin_options_zipmap
    auto_tutorial/plot_dbegin_options_list
    auto_tutorial/plot_ebegin_float_double
    auto_tutorial/plot_fbegin_investigate
    auto_tutorial/plot_gbegin_cst
    auto_tutorial/plot_gbegin_dataframe
    auto_tutorial/plot_gconverting
