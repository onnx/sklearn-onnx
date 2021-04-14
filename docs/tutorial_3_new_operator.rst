..  SPDX-License-Identifier: Apache-2.0


Extend ONNX, extend runtime
===========================

Existing converters assume it is possible to convert
a model with the current list of :epkg:`ONNX operators`.
This list is growing at every version but it may happen
a new node is needed. It could be added to ONNX specifications,
it requires a new release, but that's not mandatory.
New nodes can easily be created by using a different domain.
A domain defines a set of operators, there are currently two
officially supported domains: :epkg:`ONNX operators` and
:epkg:`ONNX ML operators`. Custom domains can be used.
Once this new node is defined, a converter can use it.
That leaves the last issue: the runtime must be aware
of the implementation attached to this new node.
That's the difficult part.

.. toctree::
    :maxdepth: 1

    auto_tutorial/plot_pextend_python_runtime
    auto_tutorial/plot_qextend_onnxruntime
