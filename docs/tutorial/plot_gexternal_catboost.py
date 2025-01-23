# SPDX-License-Identifier: Apache-2.0

"""
.. _example-catboost:

Convert a pipeline with a CatBoost classifier
=============================================

.. index:: CatBoost

:epkg:`sklearn-onnx` only converts :epkg:`scikit-learn` models into *ONNX*
but many libraries implement :epkg:`scikit-learn` API so that their models
can be included in a :epkg:`scikit-learn` pipeline. This example considers
a pipeline including a :epkg:`CatBoost` model. :epkg:`sklearn-onnx` can convert
the whole pipeline as long as it knows the converter associated to
a *CatBoostClassifier*. Let's see how to do it.

Train a CatBoostClassifier
++++++++++++++++++++++++++
"""

import numpy
from onnx.helper import get_attribute_value
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import onnxruntime as rt
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    guess_tensor_type,
)
from skl2onnx._parse import _apply_zipmap, _get_sklearn_operator_name
from catboost import CatBoostClassifier
from catboost.utils import convert_to_onnx_object

data = load_iris()
X = data.data[:, :2]
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

pipe = Pipeline(
    [("scaler", StandardScaler()), ("lgbm", CatBoostClassifier(n_estimators=3))]
)
pipe.fit(X, y)

######################################
# Register the converter for CatBoostClassifier
# +++++++++++++++++++++++++++++++++++++++++++++
#
# The model has no converter implemented in sklearn-onnx.
# We need to register the one coming from *CatBoost* itself.
# However, the converter does not follow sklearn-onnx design and
# needs to be wrapped.


def skl2onnx_parser_castboost_classifier(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(zipmap=True))
    no_zipmap = isinstance(options["zipmap"], bool) and not options["zipmap"]

    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs

    label_variable = scope.declare_local_variable("label", Int64TensorType())
    prob_dtype = guess_tensor_type(inputs[0].type)
    probability_tensor_variable = scope.declare_local_variable(
        "probabilities", prob_dtype
    )
    this_operator.outputs.append(label_variable)
    this_operator.outputs.append(probability_tensor_variable)
    probability_tensor = this_operator.outputs

    if no_zipmap:
        return probability_tensor

    return _apply_zipmap(
        options["zipmap"], scope, model, inputs[0].type, probability_tensor
    )


def skl2onnx_convert_catboost(scope, operator, container):
    """
    CatBoost returns an ONNX graph with a single node.
    This function adds it to the main graph.
    """
    onx = convert_to_onnx_object(operator.raw_operator)
    opsets = {d.domain: d.version for d in onx.opset_import}
    if "" in opsets and opsets[""] >= container.target_opset:
        raise RuntimeError("CatBoost uses an opset more recent than the target one.")
    if len(onx.graph.initializer) > 0 or len(onx.graph.sparse_initializer) > 0:
        raise NotImplementedError(
            "CatBoost returns a model initializers. This option is not implemented yet."
        )
    if (
        len(onx.graph.node) not in (1, 2)
        or not onx.graph.node[0].op_type.startswith("TreeEnsemble")
        or (len(onx.graph.node) == 2 and onx.graph.node[1].op_type != "ZipMap")
    ):
        types = ", ".join(map(lambda n: n.op_type, onx.graph.node))
        raise NotImplementedError(
            f"CatBoost returns {len(onx.graph.node)} != 1 (types={types}). "
            f"This option is not implemented yet."
        )
    node = onx.graph.node[0]
    atts = {}
    for att in node.attribute:
        atts[att.name] = get_attribute_value(att)
    container.add_node(
        node.op_type,
        [operator.inputs[0].full_name],
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain=node.domain,
        op_version=opsets.get(node.domain, None),
        **atts,
    )


update_registered_converter(
    CatBoostClassifier,
    "CatBoostCatBoostClassifier",
    calculate_linear_classifier_output_shapes,
    skl2onnx_convert_catboost,
    parser=skl2onnx_parser_castboost_classifier,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)

##################################
# Convert
# +++++++

model_onnx = convert_sklearn(
    pipe,
    "pipeline_catboost",
    [("input", FloatTensorType([None, 2]))],
    target_opset={"": 12, "ai.onnx.ml": 2},
)

# And save.
with open("pipeline_catboost.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

###########################
# Compare the predictions
# +++++++++++++++++++++++
#
# Predictions with CatBoost.

print("predict", pipe.predict(X[:5]))
print("predict_proba", pipe.predict_proba(X[:1]))

##########################
# Predictions with onnxruntime.

sess = rt.InferenceSession("pipeline_catboost.onnx", providers=["CPUExecutionProvider"])

pred_onx = sess.run(None, {"input": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])
