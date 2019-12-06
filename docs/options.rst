
================
Specific options
================

Some models can be converted into ONNX in multiple ways.
That's why there is for some model not one corresponding
ONNX graph but several the user can choose by giving 
additiona parameters to functions
:func:`convert_sklearn <skl2onnx.convert_sklearn>`
or :func:`to_onnx <skl2onnx.to_onnx>`.

.. contents::
    :local:

GaussianProcessRegressor, NearestNeighbors
==========================================

.. index:: pairwise distances, cdist

Both models require to compure pairwise distances.
Function :func:`onnx_cdist <skl2onnx.algebra.complex_functions.onnx_cdist>`
produces this part of the graph but there exist two options.
The first one is using *Scan* operator, the scond one is
using a dedicated operator called *CDist*.

