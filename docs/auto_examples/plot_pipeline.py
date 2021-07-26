# SPDX-License-Identifier: Apache-2.0


"""
Draw a pipeline
===============

There is no other way to look into one model stored
in ONNX format than looking into its node with
*onnx*. This example demonstrates
how to draw a model and to retrieve it in *json*
format.

.. contents::
    :local:

Retrieve a model in JSON format
+++++++++++++++++++++++++++++++

That's the most simple way.
"""

import skl2onnx
import onnxruntime
import sklearn
import numpy
import matplotlib.pyplot as plt
import os
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
from onnx import ModelProto
import onnx
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxMul

onnx_fct = OnnxAdd(
    OnnxMul('X', numpy.array([2], dtype=numpy.float32),
            op_version=12),
    numpy.array([[1, 0], [0, 1]], dtype=numpy.float32),
    output_names=['Y'], op_version=12)

X = numpy.array([[4, 5], [-2, 3]], dtype=numpy.float32)
model = onnx_fct.to_onnx({'X': X}, target_opset=12)
print(model)

filename = "example1.onnx"
with open(filename, "wb") as f:
    f.write(model.SerializeToString())


#################################
# Draw a model with ONNX
# ++++++++++++++++++++++
# We use `net_drawer.py
# <https://github.com/onnx/onnx/blob/master/onnx/tools/net_drawer.py>`_
# included in *onnx* package.
# We use *onnx* to load the model
# in a different way than before.


model = ModelProto()
with open(filename, 'rb') as fid:
    content = fid.read()
    model.ParseFromString(content)

###################################
# We convert it into a graph.
pydot_graph = GetPydotGraph(model.graph, name=model.graph.name, rankdir="TB",
                            node_producer=GetOpNodeProducer("docstring"))
pydot_graph.write_dot("graph.dot")

#######################################
# Then into an image
os.system('dot -O -Tpng graph.dot')

################################
# Which we display...
image = plt.imread("graph.dot.png")
plt.imshow(image)
plt.axis('off')

#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
