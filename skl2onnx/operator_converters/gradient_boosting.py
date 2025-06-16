# SPDX-License-Identifier: Apache-2.0


import numbers
import numpy as np
from ..common._apply_operation import apply_cast
from ..common.data_types import BooleanTensorType, Int64TensorType, guess_numpy_type
from ..common._registration import register_converter
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_classifier_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
)
from ..proto import onnx_proto


def convert_sklearn_gradient_boosting_classifier(
        scope,
        operator,
        container,
        op_type="TreeEnsembleClassifier",
        op_domain="ai.onnx.ml",
        op_version=1,
):
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    op = operator.raw_operator
    if op.loss not in ("deviance", "log_loss", "exponential"):
        raise NotImplementedError(
            "Loss '{0}' is not supported yet. You "
            "may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.".format(op.loss)
        )

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs["name"] = scope.get_unique_operator_name(op_type)

    transform = "LOGISTIC" if op.n_classes_ == 2 else "SOFTMAX"
    options = container.get_options(op, dict(raw_scores=False))

    if op.init == "zero":
        loss = op._loss if hasattr(op, "_loss") else op.loss_
        if hasattr(loss, "K"):
            base_values = np.zeros(loss.K)
        else:
            base_values = np.zeros(1)
    elif op.init is None:
        if hasattr(op.estimators_[0, 0], "n_features_in_"):
            # sklearn >= 1.2
            n_features = op.estimators_[0, 0].n_features_in_
        else:
            # sklearn < 1.2
            n_features = op.estimators_[0, 0].n_features_
        x0 = np.zeros((1, n_features))
        if hasattr(op, "_raw_predict_init"):
            # sklearn >= 0.21
            base_values = op._raw_predict_init(x0).ravel()
        elif hasattr(op, "_init_decision_function"):
            # sklearn >= 0.20 and sklearn < 0.21
            base_values = op._init_decision_function(x0).ravel()
        else:
            raise RuntimeError("scikit-learn < 0.19 is not supported.")
    else:
        raise NotImplementedError(
            "Setting init to an estimator is not supported, you may raise an "
            "issue at https://github.com/onnx/sklearn-onnx/issues."
        )

    if op.loss == "exponential":
        attrs["post_transform"] = "NONE"
        apply_exponential_sigmoid_patch = True
    else:
        apply_exponential_sigmoid_patch = False
        if not options["raw_scores"]:
            attrs["post_transform"] = transform

    attrs["base_values"] = [float(v) for v in base_values]

    classes = op.classes_
    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        attrs["classlabels_int64s"] = class_labels
    elif all(isinstance(i, str) for i in classes):
        class_labels = [str(i) for i in classes]
        attrs["classlabels_strings"] = class_labels
    else:
        raise ValueError("Labels must be all integer or all strings.")

    tree_weight = op.learning_rate
    n_est = op.n_estimators_ if hasattr(op, "n_estimators_") else op.n_estimators
    if op.n_classes_ == 2:
        for tree_id in range(n_est):
            tree = op.estimators_[tree_id][0].tree_
            add_tree_to_attribute_pairs(
                attrs, True, tree, tree_id, tree_weight, 0, False, True, dtype=dtype
            )
    else:
        for i in range(n_est):
            for c in range(op.n_classes_):
                tree_id = i * op.n_classes_ + c
                tree = op.estimators_[i][c].tree_
                add_tree_to_attribute_pairs(
                    attrs, True, tree, tree_id, tree_weight, c, False, True, dtype=dtype
                )

    if dtype is not None:
        for k in attrs:
            if k in (
                    "nodes_values",
                    "class_weights",
                    "target_weights",
                    "nodes_hitrates",
                    "base_values",
            ):
                attrs[k] = np.array(attrs[k], dtype=dtype)

    input_name = operator.input_full_names
    if isinstance(operator.inputs[0].type, BooleanTensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")

        apply_cast(
            scope,
            input_name,
            cast_input_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        input_name = cast_input_name

    if apply_exponential_sigmoid_patch:
        raw_proba_output = scope.get_unique_variable_name("raw_logits")
    else:
        raw_proba_output = operator.outputs[1].full_name

    container.add_node(
        op_type,
        input_name,
        [operator.outputs[0].full_name, raw_proba_output],
        op_domain=op_domain,
        op_version=op_version,
        **attrs,
    )

    if apply_exponential_sigmoid_patch:
        from ..algebra.onnx_ops import OnnxMul, OnnxSigmoid

        scale_name = scope.get_unique_variable_name("scale2")
        scaled_logits = scope.get_unique_variable_name("scaled_logits")

        container.add_initializer(scale_name, onnx_proto.TensorProto.FLOAT, [], [2.0])
        container.add_node(
            "Mul",
            [raw_proba_output, scale_name],
            scaled_logits,
            name=scope.get_unique_operator_name("Mul_Exp2"),
        )
        container.add_node(
            "Sigmoid",
            [scaled_logits],
            operator.outputs[1].full_name,
            name=scope.get_unique_operator_name("Sigmoid_Exp2"),
        )


def convert_sklearn_gradient_boosting_regressor(
        scope,
        operator,
        container,
        op_type="TreeEnsembleRegressor",
        op_domain="ai.onnx.ml",
        op_version=1,
):
    op = operator.raw_operator
    attrs = get_default_tree_regressor_attribute_pairs()
    attrs["name"] = scope.get_unique_operator_name(op_type)
    attrs["n_targets"] = 1

    if op.init == "zero":
        loss = op._loss if hasattr(op, "_loss") else op.loss_
        if hasattr(loss, "K"):
            cst = np.zeros(loss.K)
        else:
            cst = np.zeros(1)
    elif op.init is None:
        # constant_ was introduced in scikit-learn 0.21.
        if hasattr(op.init_, "constant_"):
            if len(op.init_.constant_.shape) == 0:
                cst = [float(op.init_.constant_)]
            else:
                cst = [float(x) for x in op.init_.constant_.ravel().tolist()]
        elif op.loss == "ls":
            cst = [op.init_.mean]
        else:
            cst = [op.init_.quantile]
    else:
        raise NotImplementedError(
            "Setting init to an estimator is not supported, you may raise an "
            "issue at https://github.com/onnx/sklearn-onnx/issues."
        )

    attrs["base_values"] = [float(x) for x in cst]

    tree_weight = op.learning_rate
    n_est = op.n_estimators_ if hasattr(op, "n_estimators_") else op.n_estimators
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    for i in range(n_est):
        tree = op.estimators_[i][0].tree_
        tree_id = i
        add_tree_to_attribute_pairs(
            attrs, False, tree, tree_id, tree_weight, 0, False, True, dtype=dtype
        )

    if dtype is not None:
        for k in attrs:
            if k in (
                    "nodes_values",
                    "class_weights",
                    "target_weights",
                    "nodes_hitrates",
                    "base_values",
            ):
                attrs[k] = np.array(attrs[k], dtype=dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")

        apply_cast(
            scope,
            operator.input_full_names,
            cast_input_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        input_name = cast_input_name

    container.add_node(
        op_type,
        input_name,
        operator.output_full_names,
        op_domain=op_domain,
        op_version=op_version,
        **attrs,
    )


register_converter(
    "SklearnGradientBoostingClassifier",
    convert_sklearn_gradient_boosting_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "raw_scores": [True, False],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
register_converter(
    "SklearnGradientBoostingRegressor", convert_sklearn_gradient_boosting_regressor
)
