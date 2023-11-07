import logging
import numpy as np

from ..helpers.dict_wrapper import DictWrapper, fill_missing_indices
from .._supported_operators import get_model_alias
from ..proto import onnx_proto
from ..common._apply_operation import apply_identity
from ..common.utils import check_input_and_output_types
from ..common.data_types import FloatTensorType, Int64TensorType, guess_numpy_type
from ..algebra.onnx_operator import OnnxSubEstimator
from ..algebra.onnx_ops import (
    OnnxSqueeze,
    OnnxSlice,
    OnnxIdentity,
    OnnxDiv,
    OnnxArgMax,
    OnnxAdd,
)

# instantiate the logger
file_handler = logging.FileHandler("prog2onnx.log", delay=True, mode="a")
stdout_handler = logging.StreamHandler()
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    datefmt="%d/%m/%y %H:%M:%S",
    format="[%(asctime)s] - %(name)17s | %(levelname)-7s: %(message)2s",
    handlers=handlers,
)

logger = logging.getLogger("prog2onnx")
logger.setLevel(logging.WARN)  # set level to warn (less verbose output)


def prog_transformer_converter(scope, operator, container):
    """
    Converts a progressive transformer operator to ONNX format.

    Parameters
    ----------
    scope : Scope
        The scope object for the current model.
    operator : Operator
        The operator object representing the progressive transformer operator.
    container : ModelComponentContainer
        The container object for the current model.

    Notes
    -----
    This function checks the input and output types of the operator, retrieves the task ID from the raw operator,
    and determines whether the default voter class is a tree voter or a KNN voter. It then creates an ONNX subgraph
    for each estimator in the raw operator, adds them to the total probability, and adds nodes to the container to
    calculate the final probability and anomaly score. If the default voter class is not recognized, it raises a
    ValueError.
    """
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)

    if hasattr(op, "task_id"):
        task_id = op.task_id
        logger.info("Setting task_id = %d" % task_id)
    else:
        raise AttributeError(
            "`task_id` attribute not found in `ClassificationProgressiveLearner` object"
        )

    tree_voter = True
    voters = np.asarray(
        list(op.task_id_to_transformer_id_to_voters[task_id].values())
    ).flatten()
    n_fitted = list(op.task_id_to_y.values())[-1].size
    logger.info("Each transformer fitted using %d training instances" % n_fitted)
    if op.default_voter_class.__name__ == "TreeClassificationVoter":
        max_keys = max(list(map(lambda x: max(x.leaf_to_posterior_.keys()), voters)))
        posterior_length = len(list(voters[0].leaf_to_posterior_.values())[0])
        scale_ = (n_fitted // max_keys) + 1
        logger.info("Max key found: %d" % max_keys)
        logger.info("Number of classes detected: %d" % posterior_length)
        logger.info(
            "Adjusting `scale` parameter to match number of training instances (T:%d) (A:%d)"
            % (n_fitted, (scale_ * max_keys))
        )
    elif op.default_voter_class.__name__ == "KNNClassificationVoter":
        tree_voter ^= tree_voter
    else:
        raise ValueError("Unknown voter class: %r" % op.default_voter_class.__name__)

    logger.info("Voter class detected: %r" % op.default_voter_class.__name__)
    logger.info(
        "Transformer class detected: %r " % op.default_transformer_class.__name__
    )
    n_classes = np.unique(op.task_id_to_y[task_id]).size
    total_prob = np.asarray([[n_classes * [0.0]]], dtype=dtype)
    transformers = np.asarray(
        list(op.transformer_id_to_transformers.values())
    ).flatten()
    estimators = transformers.copy()
    for i, estimator in enumerate(transformers):
        if tree_voter:
            est = OnnxSubEstimator(
                estimator.transformer_,
                X,
                op_version=opv,
                options={"decision_leaf": True, "zipmap": False},
            )
            leaf = OnnxIdentity(est[2], op_version=opv)
            posterior = OnnxSubEstimator(
                DictWrapper(
                    fill_missing_indices(
                        voters[i].leaf_to_posterior_,
                        max_keys,
                        posterior_len=posterior_length,
                        scale=scale_,
                    )
                ),
                leaf,
                op_version=opv,
            )
            posterior_val = OnnxIdentity(posterior, op_version=opv)
        else:
            raise NotImplementedError("`KNNClassificationVoter` not implemented.")

        total_prob = OnnxAdd(posterior_val, total_prob)

    logger.info("Total number of estimators: %d" % estimators.size)
    if tree_voter:
        total_prob_final = OnnxDiv(
            total_prob,
            np.asarray([len(estimators)], dtype=dtype),
            op_version=opv,
        )
        total_prob_sliced = OnnxSlice(
            total_prob_final,
            np.array([1]),
            np.array([2]),
            np.array([0]),
            op_version=opv,
        )
        total_prob_final_sq = OnnxSqueeze(
            total_prob_sliced,
            op_version=opv,
            output_names=out[1],
        )
        anomaly = OnnxArgMax(total_prob_sliced, axis=2)
    else:
        pass

    total_prob_final_sq.add_to(scope, container)
    logger.info("Added predicted class probabilities to ONNX graph")
    anomaly_flat = OnnxSqueeze(anomaly, output_names=out[:1])
    anomaly_flat.add_to(scope, container)
    logger.info("Added classification labels to ONNX graph")


def progresive_parser(scope, model, inputs, custom_parsers=None):
    """
    Parses a progressive learning model to a format compatible with ONNX.

    Parameters
    ----------
    scope : Scope
        The scope object for the current model.
    model : object
        The progressive learning model to be parsed.
    inputs : list
        The list of input variables for the model.
    custom_parsers : dict, default=None
        A dictionary of custom parsers to use for specific types of models.

    Returns
    -------
    list
        The list of output variables for the model.

    Notes
    -----
    This function declares a local operator for the model and adds the input variables to it. It then declares two
    local variables for the outputs and adds them to the operator. The output variables are logged and returned.
    """
    alias = get_model_alias(type(model))
    this_operator = scope.declare_local_operator(alias, model)

    # inputs
    this_operator.inputs.append(inputs[0])
    cls_type = inputs[0].type.__class__

    # outputs
    val_y1 = scope.declare_local_variable("class", Int64TensorType())
    val_y2 = scope.declare_local_variable("probability", cls_type())

    this_operator.outputs.append(val_y1)
    this_operator.outputs.append(val_y2)
    logger.info("Added output (1. %s)" % val_y1.full_name)
    logger.info("Added output (2. %s)" % val_y2.full_name)

    return this_operator.outputs


def dict_custom_converter(scope, operator, container):
    """
    Converts a custom dictionary operator to ONNX format.

    Parameters
    ----------
    scope : Scope
        The scope object for the current model.
    operator : Operator
        The operator object representing the custom dictionary operator.
    container : ModelComponentContainer
        The container object for the current model.

    Notes
    -----
    This function adds initializers and nodes to the container object to represent
    the custom dictionary operator. It uses the unique variable names generated by
    the scope object and the data from the operator object.
    """
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    dict_tensor = scope.get_unique_variable_name("dict_tensor")
    dict_values = np.asarray(list(op.data.values()), dtype=np.float32)
    container.add_initializer(
        dict_tensor,
        onnx_proto.TensorProto.FLOAT,
        dict_values.shape,
        dict_values.flatten(),
    )

    dict_keys = scope.get_unique_variable_name("dict_keys")
    container.add_initializer(
        dict_keys,
        onnx_proto.TensorProto.INT64,
        [len(op.data)],
        list(op.data.keys()),
    )

    equal_name = scope.get_unique_variable_name("equal")
    container.add_node(
        "Equal",
        [dict_keys, operator.inputs[0].full_name],
        equal_name,
        op_version=opv,
    )

    nonzero_name = scope.get_unique_variable_name("nonzero")
    container.add_node("NonZero", [equal_name], nonzero_name, op_version=opv)

    value_name = scope.get_unique_variable_name("value")
    container.add_node(
        "Gather", [dict_tensor, nonzero_name], value_name, op_version=opv
    )

    new_shape_value = scope.get_unique_variable_name("new_shape_value")
    container.add_initializer(new_shape_value, onnx_proto.TensorProto.INT64, [1], [-1])

    new_shape_1d = scope.get_unique_variable_name("new_shape_1d")
    container.add_node("Reshape", [value_name, new_shape_value], [new_shape_1d])

    apply_identity(scope, value_name, out[0].full_name, container)
