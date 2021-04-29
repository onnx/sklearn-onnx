# SPDX-License-Identifier: Apache-2.0


"""
.. _l-custom-parser-alternative:

When a custom model is neither a classifier nor a regressor (alternative)
=========================================================================

.. note::
    This example rewrites :ref:`l-custom-parser` by using
    the syntax proposed in example :ref:`l-onnx-operators`
    to write the custom converter, shape calculator and parser.

*scikit-learn*'s API specifies that a regressor produces one
outputs and a classifier produces two
outputs, predicted labels and probabilities. The goal here is
to add a third result which tells if the probability is
above a given threshold. That's implemented in method
*validate*.

.. contents::
    :local:

Iris and scoring
++++++++++++++++

A new class is created, it trains any classifier and implements
the method *validate* mentioned above.
"""
import inspect
import numpy as np
import skl2onnx
import onnx
import sklearn
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skl2onnx import update_registered_converter
import os
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnxruntime as rt
from skl2onnx import to_onnx, get_model_alias
from skl2onnx.proto import onnx_proto
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.algebra.onnx_ops import (
    OnnxGreater, OnnxCast, OnnxReduceMax, OnnxIdentity
)
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
import matplotlib.pyplot as plt


class ValidatorClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator=None, threshold=0.75):
        ClassifierMixin.__init__(self)
        BaseEstimator.__init__(self)
        if estimator is None:
            estimator = LogisticRegression(solver='liblinear')
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None):
        sig = inspect.signature(self.estimator.fit)
        if 'sample_weight' in sig.parameters:
            self.estimator_ = clone(self.estimator).fit(
                X, y, sample_weight=sample_weight)
        else:
            self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def validate(self, X):
        pred = self.predict_proba(X)
        mx = pred.max(axis=1)
        return (mx >= self.threshold) * 1


data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = ValidatorClassifier()
model.fit(X_train, y_train)

##############################
# Let's now measure the indicator which tells
# if the probability of a prediction is above
# a threshold.

print(model.validate(X_test))

####################################
# Conversion to ONNX
# +++++++++++++++++++
#
# The conversion fails for a new model because
# the library does not know any converter associated
# to this new model.

try:
    to_onnx(model, X_train[:1].astype(np.float32),
            target_opset=12)
except RuntimeError as e:
    print(e)

############################################
# Custom converter
# ++++++++++++++++
#
# We reuse some pieces of code from :ref:`l-custom-model`.
# The shape calculator defines the shape of every output
# of the converted model.


def validator_classifier_shape_calculator(operator):

    input0 = operator.inputs[0]     # first input in ONNX graph
    outputs = operator.outputs      # outputs in ONNX graph
    op = operator.raw_operator      # scikit-learn model (mmust be fitted)
    if len(outputs) != 3:
        raise RuntimeError("3 outputs expected not {}.".format(len(outputs)))

    N = input0.type.shape[0]                    # number of observations
    C = op.estimator_.classes_.shape[0]         # dimension of outputs

    outputs[0].type = Int64TensorType([N])      # label
    outputs[1].type = FloatTensorType([N, C])   # probabilities
    outputs[2].type = Int64TensorType([C])      # validation

#############################
# Then the converter.


def validator_classifier_converter(scope, operator, container):
    input0 = operator.inputs[0]         # first input in ONNX graph
    outputs = operator.outputs          # outputs in ONNX graph
    op = operator.raw_operator          # scikit-learn model (mmust be fitted)
    opv = container.target_opset

    # The model calls another one. The class `OnnxSubEstimator`
    # calls the converter for this operator.
    model = op.estimator_
    onnx_op = OnnxSubEstimator(model, input0, op_version=opv,
                               options={'zipmap': False})

    rmax = OnnxReduceMax(onnx_op[1], axes=[1], keepdims=0, op_version=opv)
    great = OnnxGreater(rmax, np.array([op.threshold], dtype=np.float32),
                        op_version=opv)
    valid = OnnxCast(great, to=onnx_proto.TensorProto.INT64,
                     op_version=opv)

    r1 = OnnxIdentity(onnx_op[0], output_names=[outputs[0].full_name],
                      op_version=opv)
    r2 = OnnxIdentity(onnx_op[1], output_names=[outputs[1].full_name],
                      op_version=opv)
    r3 = OnnxIdentity(valid, output_names=[outputs[2].full_name],
                      op_version=opv)

    r1.add_to(scope, container)
    r2.add_to(scope, container)
    r3.add_to(scope, container)


##########################
# Then the registration.


update_registered_converter(ValidatorClassifier, 'CustomValidatorClassifier',
                            validator_classifier_shape_calculator,
                            validator_classifier_converter)

########################
# And conversion...

try:
    to_onnx(model, X_test[:1].astype(np.float32),
            target_opset=12)
except RuntimeError as e:
    print(e)

#######################################
# It fails because the library expected the model
# to behave like a classifier which produces two
# outputs. We need to add a custom parser to
# tell the library this model produces three outputs.
#
# Custom parser
# +++++++++++++


def validator_classifier_parser(scope, model, inputs, custom_parsers=None):
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)

    # inputs
    this_operator.inputs.append(inputs[0])

    # outputs
    val_label = scope.declare_local_variable('val_label', Int64TensorType())
    val_prob = scope.declare_local_variable('val_prob', FloatTensorType())
    val_val = scope.declare_local_variable('val_val', Int64TensorType())
    this_operator.outputs.append(val_label)
    this_operator.outputs.append(val_prob)
    this_operator.outputs.append(val_val)

    # ends
    return this_operator.outputs

###############################
# Registration.


update_registered_converter(ValidatorClassifier, 'CustomValidatorClassifier',
                            validator_classifier_shape_calculator,
                            validator_classifier_converter,
                            parser=validator_classifier_parser)

#############################
# And conversion again.

model_onnx = to_onnx(model, X_test[:1].astype(np.float32),
                     target_opset=12)

#######################################
# Final test
# ++++++++++
#
# We need now to check the results are the same with ONNX.

X32 = X_test[:5].astype(np.float32)

sess = rt.InferenceSession(model_onnx.SerializeToString())
results = sess.run(None, {'X': X32})

print("--labels--")
print("sklearn", model.predict(X32))
print("onnx", results[0])
print("--probabilities--")
print("sklearn", model.predict_proba(X32))
print("onnx", results[1])
print("--validation--")
print("sklearn", model.validate(X32))
print("onnx", results[2])

##################################
# It looks good.
#
# Display the ONNX graph
# ++++++++++++++++++++++

pydot_graph = GetPydotGraph(
    model_onnx.graph, name=model_onnx.graph.name, rankdir="TB",
    node_producer=GetOpNodeProducer(
        "docstring", color="yellow", fillcolor="yellow", style="filled"))
pydot_graph.write_dot("validator_classifier.dot")

os.system('dot -O -Gdpi=300 -Tpng validator_classifier.dot')

image = plt.imread("validator_classifier.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')

#################################
# **Versions used for this example**

print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
