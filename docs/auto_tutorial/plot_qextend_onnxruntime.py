# SPDX-License-Identifier: Apache-2.0

"""
Fast runtime with onnxruntime
=============================

:epkg:`ONNX operators` does not contain operator
from :epkg:`numpy`. There is no operator for
`solve <https://numpy.org/doc/stable/reference/
generated/numpy.linalg.solve.html>`_ but this one
is needed to implement the prediction function
of model :epkg:`NMF`. The converter can be written
including a new ONNX operator but then it requires a
runtime for it to be tested. Example
:ref:`l-extend-python-runtime` shows how to do that
with :epkg:`mlprodict`. Doing the same with
:epkg:`onnxruntime` is more ambitious as it requires
C++...

*to be continued*
"""
