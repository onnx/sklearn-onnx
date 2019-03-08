
===========
API Summary
===========

Summary of public functions and classes exposed
in *scikit-onnx*.

.. contents::
    :local:

Converters
==========

.. autofunction:: skl2onnx.convert_sklearn

Manipulate ONNX graphs
======================

.. autofunction:: skl2onnx.helpers.onnx_helper.enumerate_model_node_outputs

.. autofunction:: skl2onnx.helpers.onnx_helper.load_onnx_model

.. autofunction:: skl2onnx.helpers.onnx_helper.select_model_inputs_outputs

.. autofunction:: skl2onnx.helpers.onnx_helper.save_onnx_model

Registered functions
====================

.. autofunction:: skl2onnx.supported_converters

.. autofunction:: skl2onnx.update_registered_converter

.. autofunction:: skl2onnx.update_registered_parser

Parsers
=======

.. autofunction:: skl2onnx._parse.parse_sklearn

.. autofunction:: skl2onnx._parse.parse_sklearn_model


Utils for contributors
======================

.. autofunction:: skl2onnx.common.utils.check_input_and_output_numbers

.. autofunction:: skl2onnx.common.utils.check_input_and_output_types


Concepts
========

Containers
----------

.. autoclass:: skl2onnx.common._container.SklearnModelContainerNode
    :members: input_names, output_names, add_input, add_output

.. autoclass:: skl2onnx.common._container.ModelComponentContainer
    :members: add_input, add_output, add_initializer, add_node

Nodes
-----

.. autoclass:: skl2onnx.common._topology.Operator

.. autoclass:: skl2onnx.common._topology.Variable

Scope
-----

.. autoclass:: skl2onnx.common._topology.Scope
    :members: get_unique_variable_name, get_unique_operator_name

Topology
--------

.. autoclass:: skl2onnx.common._topology.Topology
    :members: compile

