..  SPDX-License-Identifier: Apache-2.0


===========
API Summary
===========

Summary of public functions and classes exposed
in *scikit-onnx*.

Version
=======

.. autofunction:: skl2onnx.get_latest_tested_opset_version

Converters
==========

Both functions convert a *scikit-learn* model into ONNX.
The first one lets the user manually
define the input's name and types. The second one
infers this information from the training data.
These two functions are the main entry points to converter.
The rest of the API is needed if a model has no converter
implemented in this package. A new converter has then to be
registered, whether it is imported from another package
or created from scratch.

.. autofunction:: skl2onnx.convert_sklearn

.. autofunction:: skl2onnx.to_onnx

Logging
=======

.. index:: logging

The conversion of a pipeline fails if it contains an object without any
associated converter. It may also fails if one of the object is mapped
by a custom converter. If the error message is not explicit enough,
it is possible to enable logging:

::

    import logging
    logger = logging.getLogger('skl2onnx')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

Example :ref:`l-example-logging` illustrates what it looks like.

Register a new converter
========================

If a model has no converter
implemented in this package, a new converter has then to be
registered, whether it is imported from another package
or created from scratch. Section :ref:`l-converter-list`
lists all available converters.

.. autofunction:: skl2onnx.supported_converters

.. autofunction:: skl2onnx.update_registered_converter

.. autofunction:: skl2onnx.update_registered_parser

Manipulate ONNX graphs
======================

.. autofunction:: skl2onnx.helpers.onnx_helper.enumerate_model_node_outputs

.. autofunction:: skl2onnx.helpers.onnx_helper.load_onnx_model

.. autofunction:: skl2onnx.helpers.onnx_helper.select_model_inputs_outputs

.. autofunction:: skl2onnx.helpers.onnx_helper.save_onnx_model

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
    :members: compile, topological_operator_iterator

.. autofunction:: skl2onnx.common._topology.convert_topology
