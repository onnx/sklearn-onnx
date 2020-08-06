
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

    auto_examples/plot_abegin_convert_pipeline
    auto_examples/plot_bbegin_measure_time
    auto_examples/plot_cbegin_opset
    auto_examples/plot_dbegin_options
    auto_examples/plot_dbegin_options_list
    auto_examples/plot_ebegin_float_double
    auto_examples/plot_fbegin_investigate
    auto_examples/plot_gbegin_dataframe
    auto_examples/plot_gbegin_transfer_learning
