from ..common.data_types import FloatTensorType


def prog_transformer_shape_calculator(operator):
    """
    Calculates the output shape for a progressive transformer operator.

    Parameters
    ----------
    operator : Operator
        The operator object representing the progressive transformer operator.

    Notes
    -----
    This function sets the shape of the operator's outputs based on the size of the default decider in the operator's
    raw operator and the first dimension of the operator's inputs.
    """
    out_ft = list(operator.raw_operator.default_decider_kwargs.values())[0].size
    N = operator.inputs[0].get_first_dimension()

    operator.outputs[0].type.shape = [N]
    operator.outputs[1].type.shape = [N, out_ft]


def dict_shape_calculator(operator):
    """
    Calculates the output shape for a custom dictionary operator.

    Parameters
    ----------
    operator : Operator
        The operator object representing the custom dictionary operator.

    Notes
    -----
    This function sets the type of the operator's output to a float tensor with a shape
    determined by the length of the operator's data.
    """
    op = operator.raw_operator
    operator.outputs[0].type = FloatTensorType([len(op)])
