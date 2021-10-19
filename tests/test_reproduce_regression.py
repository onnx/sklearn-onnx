from typing import List

import numpy as np
import pandas as pd
from skl2onnx import get_model_alias, to_onnx, update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from skl2onnx.common._topology import Operator, Scope, Variable


class IdentityTransformer:
    pass


def parser(
    scope: Scope,
    model: IdentityTransformer,
    inputs: List[Variable],
    custom_parsers=None,
) -> List[Variable]:
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs.append(inputs[0])
    cls_type = inputs[0].type.__class__

    # If we make the output name unique, the input name is not renamed
    # and the test passes.
    val_y1 = scope.declare_local_variable(inputs[0].onnx_name, cls_type(), rename=True)
    this_operator.outputs.append(val_y1)
    return this_operator.outputs


def shape_calculator(operator: Operator):
    input_type = operator.inputs[0].type.__class__
    # The shape may be unknown. *get_first_dimension*
    # returns the appropriate value, None in most cases
    # meaning the transformer can process any batch of observations.
    input_dim = operator.inputs[0].get_first_dimension()
    output_type = input_type([input_dim])
    operator.outputs[0].type = output_type


def converter(scope, operator, container):
    opv = container.target_opset
    out = operator.outputs
    OnnxIdentity(operator.inputs[0], op_version=opv, output_names=out[:1]).add_to(
        scope, container
    )


update_registered_converter(
    IdentityTransformer, "IdentityThing", shape_calculator, converter, parser=parser
)


def test_thing():
    id_transformer = IdentityTransformer()

    X = pd.DataFrame({"input": [1]})
    onx = to_onnx(id_transformer, X)
    # I would expect the input to match the column name
    assert onx.graph.input[0].name == "input"
    # IMHO it should be the output that is renamed, not the input?
    assert onx.graph.output[0].name == "input"
