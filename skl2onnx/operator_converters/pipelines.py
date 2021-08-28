# SPDX-License-Identifier: Apache-2.0


from ..common._apply_operation import apply_concat, apply_cast
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from .._parse import _parse_sklearn


def convert_pipeline(scope: Scope, operator: Operator,
                     container: ModelComponentContainer):
    model = operator.raw_operator
    inputs = operator.inputs
    for step in model.steps:
        step_model = step[1]
        outputs = _parse_sklearn(scope, step_model, inputs,
                                 custom_parsers=None)
        last_op = outputs.parent
        inputs = outputs
    if len(last_op.outputs) != len(operator.outputs):
        raise RuntimeError(
            "Mismatch between pipeline output %d and "
            "last step outputs %d." % (
                len(last_op.outputs), len(operator.outputs)))
        for fr, to in zip(last_op.outputs, operator.outputs):
            container.add_node(
                'Identity', fr.full_name, to.full_name,
                name=scope.get_unique_operator_name(operator.name))


def convert_feature_union(scope: Scope, operator: Operator,
                          container: ModelComponentContainer):
    raise NotImplementedError()


def convert_column_transformer(scope: Scope, operator: Operator,
                               container: ModelComponentContainer):
    raise NotImplementedError()


register_converter('SklearnPipeline', convert_pipeline)
register_converter('SklearnFeatureUnion', convert_feature_union)
register_converter('SklearnColumnTransformer', convert_column_transformer)
