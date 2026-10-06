# SPDX-License-Identifier: Apache-2.0

import pprint
import numpy as np
from onnx import TensorProto
from ..common._apply_operation import (
    apply_add,
    apply_cast,
    apply_concat,
    apply_div,
    apply_exp,
    apply_mul,
    apply_reducesum,
    apply_reshape,
    apply_softmax,
    apply_sub,
)
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import Int64TensorType, guess_proto_type
from ..common._registration import register_converter
from .._supported_operators import sklearn_operator_name_map
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None


def _transform_temperature(
    scope, container, model, scores_name, n_classes, raw_scores, proto_type
):
    """Apply scikit-learn's logit conversion and learned inverse temperature."""
    calibrators = (
        model.calibrators if hasattr(model, "calibrators") else model.calibrators_
    )
    beta_name = scope.get_unique_variable_name("temperature_beta")
    unity_name = scope.get_unique_variable_name("temperature_unity")
    container.add_initializer(beta_name, proto_type, [], [calibrators[0].beta_])
    if not (n_classes == 2 and raw_scores):
        container.add_initializer(unity_name, proto_type, [], [1.0])

    if n_classes == 2:
        # sklearn extracts the positive-class response. Binary decision scores
        # become (-s, s); binary probabilities become (1 - p, p).
        index_name = scope.get_unique_variable_name("temperature_positive_index")
        positive_name = scope.get_unique_variable_name("temperature_positive")
        negative_name = scope.get_unique_variable_name("temperature_negative")
        binary_name = scope.get_unique_variable_name("temperature_binary")
        container.add_initializer(index_name, TensorProto.INT64, [1], [1])
        container.add_node(
            "ArrayFeatureExtractor",
            [scores_name, index_name],
            positive_name,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("TemperatureAFE"),
        )
        if raw_scores:
            container.add_node(
                "Neg",
                positive_name,
                negative_name,
                name=scope.get_unique_operator_name("Neg"),
            )
        else:
            apply_sub(scope, [unity_name, positive_name], negative_name, container)
        apply_concat(
            scope, [negative_name, positive_name], binary_name, container, axis=1
        )
        scores_name = binary_name

    logits_name = scores_name
    if not (n_classes == 2 and raw_scores):
        # _convert_to_logits checks the whole batch, including multiclass
        # decision scores: values in [0, 1] and every row sum close to 1.
        zero_name = scope.get_unique_variable_name("temperature_zero")
        tolerance_name = scope.get_unique_variable_name("temperature_tolerance")
        epsilon_name = scope.get_unique_variable_name("temperature_epsilon")
        container.add_initializer(zero_name, proto_type, [], [0.0])
        container.add_initializer(tolerance_name, proto_type, [], [1e-8 + 1e-5])
        container.add_initializer(epsilon_name, proto_type, [], [1e-12])

        below_name = scope.get_unique_variable_name("temperature_below_zero")
        above_name = scope.get_unique_variable_name("temperature_above_one")
        outside_name = scope.get_unique_variable_name("temperature_outside")
        inside_name = scope.get_unique_variable_name("temperature_inside")
        sums_name = scope.get_unique_variable_name("temperature_row_sums")
        delta_name = scope.get_unique_variable_name("temperature_sum_delta")
        distance_name = scope.get_unique_variable_name("temperature_sum_distance")
        far_name = scope.get_unique_variable_name("temperature_far_from_one")
        close_name = scope.get_unique_variable_name("temperature_close_to_one")
        for op_type, inputs, output in [
            ("Less", [scores_name, zero_name], below_name),
            ("Greater", [scores_name, unity_name], above_name),
            ("Or", [below_name, above_name], outside_name),
            ("Not", [outside_name], inside_name),
        ]:
            container.add_node(
                op_type, inputs, output, name=scope.get_unique_operator_name(op_type)
            )
        apply_reducesum(scope, scores_name, sums_name, container, axes=[1])
        apply_sub(scope, [sums_name, unity_name], delta_name, container)
        for op_type, inputs, output in [
            ("Abs", [delta_name], distance_name),
            ("Greater", [distance_name, tolerance_name], far_name),
            ("Not", [far_name], close_name),
        ]:
            container.add_node(
                op_type, inputs, output, name=scope.get_unique_operator_name(op_type)
            )

        checks = []
        for check_name in [inside_name, close_name]:
            cast_name = scope.get_unique_variable_name("temperature_check")
            all_name = scope.get_unique_variable_name("temperature_all")
            bool_name = scope.get_unique_variable_name("temperature_all_bool")
            apply_cast(scope, check_name, cast_name, container, to=proto_type)
            container.add_node(
                "ReduceMin",
                cast_name,
                all_name,
                keepdims=0,
                name=scope.get_unique_operator_name("ReduceMin"),
            )
            apply_cast(scope, all_name, bool_name, container, to=TensorProto.BOOL)
            checks.append(bool_name)

        probabilities_name = scope.get_unique_variable_name("temperature_is_proba")
        safe_name = scope.get_unique_variable_name("temperature_safe_proba")
        safe_epsilon_name = scope.get_unique_variable_name("temperature_safe_epsilon")
        log_name = scope.get_unique_variable_name("temperature_log_proba")
        logits_name = scope.get_unique_variable_name("temperature_logits")
        container.add_node(
            "And",
            checks,
            probabilities_name,
            name=scope.get_unique_operator_name("And"),
        )
        # Avoid evaluating Log on negative decision scores in the unused branch.
        container.add_node(
            "Where",
            [probabilities_name, scores_name, unity_name],
            safe_name,
            name=scope.get_unique_operator_name("Where"),
        )
        apply_add(scope, [safe_name, epsilon_name], safe_epsilon_name, container)
        container.add_node(
            "Log",
            safe_epsilon_name,
            log_name,
            name=scope.get_unique_operator_name("Log"),
        )
        container.add_node(
            "Where",
            [probabilities_name, log_name, scores_name],
            logits_name,
            name=scope.get_unique_operator_name("Where"),
        )

    scaled_name = scope.get_unique_variable_name("temperature_scaled_logits")
    probabilities_name = scope.get_unique_variable_name("temperature_probabilities")
    apply_mul(scope, [beta_name, logits_name], scaled_name, container)
    apply_softmax(scope, scaled_name, probabilities_name, container, axis=1)
    return probabilities_name


def _handle_zeros(
    scope, container, concatenated_prob_name, reduced_prob_name, n_classes, proto_type
):
    """
    This function replaces 0s in concatenated_prob_name with 1s and
    0s in reduced_prob_name with n_classes.
    """
    cast_prob_name = scope.get_unique_variable_name("cast_prob")
    bool_not_cast_prob_name = scope.get_unique_variable_name("bool_not_cast_prob")
    mask_name = scope.get_unique_variable_name("mask")
    masked_concatenated_prob_name = scope.get_unique_variable_name(
        "masked_concatenated_prob"
    )
    n_classes_name = scope.get_unique_variable_name("n_classes")
    reduced_prob_mask_name = scope.get_unique_variable_name("reduced_prob_mask")
    masked_reduced_prob_name = scope.get_unique_variable_name("masked_reduced_prob")

    proto_type2 = proto_type
    if proto_type2 not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        proto_type2 = TensorProto.FLOAT

    container.add_initializer(n_classes_name, proto_type2, [], [n_classes])

    apply_cast(scope, reduced_prob_name, cast_prob_name, container, to=TensorProto.BOOL)
    container.add_node(
        "Not",
        cast_prob_name,
        bool_not_cast_prob_name,
        name=scope.get_unique_operator_name("Not"),
    )
    apply_cast(scope, bool_not_cast_prob_name, mask_name, container, to=proto_type2)
    apply_add(
        scope,
        [concatenated_prob_name, mask_name],
        masked_concatenated_prob_name,
        container,
        broadcast=1,
    )
    apply_mul(
        scope,
        [mask_name, n_classes_name],
        reduced_prob_mask_name,
        container,
        broadcast=1,
    )
    apply_add(
        scope,
        [reduced_prob_name, reduced_prob_mask_name],
        masked_reduced_prob_name,
        container,
        broadcast=0,
    )
    return masked_concatenated_prob_name, masked_reduced_prob_name


def _transform_sigmoid(scope, container, model, df_col_name, k, proto_type):
    """
    Sigmoid Calibration method
    """
    a_name = scope.get_unique_variable_name("a")
    b_name = scope.get_unique_variable_name("b")
    a_df_prod_name = scope.get_unique_variable_name("a_df_prod")
    exp_parameter_name = scope.get_unique_variable_name("exp_parameter")
    exp_result_name = scope.get_unique_variable_name("exp_result")
    unity_name = scope.get_unique_variable_name("unity")
    denominator_name = scope.get_unique_variable_name("denominator")
    sigmoid_predict_result_name = scope.get_unique_variable_name(
        "sigmoid_predict_result"
    )

    proto_type2 = proto_type
    if proto_type2 not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        proto_type2 = TensorProto.FLOAT

    if hasattr(model, "calibrators_"):
        # scikit-learn<1.1
        calibrators = model.calibrators_
    elif hasattr(model, "calibrators"):
        # scikit-learn>=1.1
        calibrators = model.calibrators
    else:
        raise AttributeError(
            "Unable to find attribute calibrators_ or "
            "calibrators, check the model was trained, "
            "type=%r." % type(model)
        )

    container.add_initializer(a_name, proto_type2, [], [calibrators[k].a_])
    container.add_initializer(b_name, proto_type2, [], [calibrators[k].b_])
    container.add_initializer(unity_name, proto_type2, [], [1])

    apply_mul(scope, [a_name, df_col_name], a_df_prod_name, container, broadcast=0)
    apply_add(
        scope, [a_df_prod_name, b_name], exp_parameter_name, container, broadcast=0
    )
    apply_exp(scope, exp_parameter_name, exp_result_name, container)
    apply_add(
        scope, [unity_name, exp_result_name], denominator_name, container, broadcast=0
    )
    apply_div(
        scope,
        [unity_name, denominator_name],
        sigmoid_predict_result_name,
        container,
        broadcast=0,
    )
    return sigmoid_predict_result_name


def _transform_isotonic(scope, container, model, T, k, proto_type):
    """
    Isotonic calibration method
    The calibrator is written as the piecewise linear function
    *IsotonicRegression.predict* interpolates between its thresholds.
    """
    if hasattr(model, "calibrators_"):
        # scikit-learn<1.1
        calibrators = model.calibrators_
    elif hasattr(model, "calibrators"):
        # scikit-learn>=1.1
        calibrators = model.calibrators
    else:
        raise AttributeError(
            "Unable to find attribute calibrators_ or "
            "calibrators, check the model was trained, "
            "type=%r." % type(model)
        )

    # thresholds can be too close together for single precision
    cast_df_name = scope.get_unique_variable_name("cast_df")
    apply_cast(scope, T, cast_df_name, container, to=TensorProto.DOUBLE)
    T = cast_df_name

    if calibrators[k].out_of_bounds == "clip":
        clip_min_name = scope.get_unique_variable_name("clip_min")
        clip_max_name = scope.get_unique_variable_name("clip_max")
        lower_df_name = scope.get_unique_variable_name("lower_df")
        clipped_df_name = scope.get_unique_variable_name("clipped_df")
        container.add_initializer(
            clip_min_name, TensorProto.DOUBLE, [], [calibrators[k].X_min_]
        )
        container.add_initializer(
            clip_max_name, TensorProto.DOUBLE, [], [calibrators[k].X_max_]
        )
        # Max and Min, Clip takes no double bounds before opset 12
        container.add_node(
            "Max",
            [T, clip_min_name],
            lower_df_name,
            name=scope.get_unique_operator_name("Max"),
        )
        container.add_node(
            "Min",
            [lower_df_name, clip_max_name],
            clipped_df_name,
            name=scope.get_unique_operator_name("Min"),
        )
        T = clipped_df_name

    reshaped_df_name = scope.get_unique_variable_name("reshaped_df")
    below_name = scope.get_unique_variable_name("below_segment_x")
    below_int_name = scope.get_unique_variable_name("below_segment_x_int")
    segment_name = scope.get_unique_variable_name("segment")

    if hasattr(calibrators[k], "_X_"):
        atX, atY = "_X_", "_y_"
    elif hasattr(calibrators[k], "_necessary_X_"):
        atX, atY = "_necessary_X_", "_necessary_y_"
    elif hasattr(calibrators[k], "X_thresholds_"):
        atX, atY = "X_thresholds_", "y_thresholds_"
    else:
        raise AttributeError(
            "Unable to find attribute '_X_' or '_necessary_X_' "
            "for type {}\n{}."
            "".format(type(calibrators[k]), pprint.pformat(dir(calibrators[k])))
        )

    proto_type2 = proto_type
    if proto_type2 not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        proto_type2 = TensorProto.FLOAT

    knots_x = np.array(getattr(calibrators[k], atX), dtype=np.float64).ravel()
    knots_y = np.array(getattr(calibrators[k], atY), dtype=np.float64).ravel()
    if len(knots_x) == 1:
        # constant calibrator, written as a flat segment
        knots_x = np.hstack([knots_x, knots_x + 1])
        knots_y = np.hstack([knots_y, knots_y])

    segment_x_name = scope.get_unique_variable_name("segment_x")
    lower_x_name = scope.get_unique_variable_name("lower_x")
    upper_x_name = scope.get_unique_variable_name("upper_x")
    lower_y_name = scope.get_unique_variable_name("lower_y")
    upper_y_name = scope.get_unique_variable_name("upper_y")

    # The segment holding T is the number of upper bounds T is greater than.
    # Repeating the last segment keeps that index in range whatever T is.
    for name, values in [
        (segment_x_name, knots_x[1:]),
        (lower_x_name, np.hstack([knots_x[:-1], knots_x[-2:-1]])),
        (upper_x_name, np.hstack([knots_x[1:], knots_x[-1:]])),
        (lower_y_name, np.hstack([knots_y[:-1], knots_y[-2:-1]])),
        (upper_y_name, np.hstack([knots_y[1:], knots_y[-1:]])),
    ]:
        container.add_initializer(name, TensorProto.DOUBLE, [len(values)], values)

    apply_reshape(scope, T, reshaped_df_name, container, desired_shape=(-1, 1))
    container.add_node(
        "Less",
        [segment_x_name, reshaped_df_name],
        below_name,
        name=scope.get_unique_operator_name("Less"),
    )
    apply_cast(scope, below_name, below_int_name, container, to=TensorProto.INT64)
    apply_reducesum(scope, below_int_name, segment_name, container, axes=[1])

    x0_name = scope.get_unique_variable_name("x0")
    x1_name = scope.get_unique_variable_name("x1")
    y0_name = scope.get_unique_variable_name("y0")
    y1_name = scope.get_unique_variable_name("y1")
    for data_name, gathered_name in [
        (lower_x_name, x0_name),
        (upper_x_name, x1_name),
        (lower_y_name, y0_name),
        (upper_y_name, y1_name),
    ]:
        container.add_node(
            "Gather",
            [data_name, segment_name],
            gathered_name,
            name=scope.get_unique_operator_name("Gather"),
        )

    delta_y_name = scope.get_unique_variable_name("delta_y")
    delta_x_name = scope.get_unique_variable_name("delta_x")
    offset_name = scope.get_unique_variable_name("offset")
    scaled_name = scope.get_unique_variable_name("scaled")
    ratio_name = scope.get_unique_variable_name("ratio")
    interpolated_name = scope.get_unique_variable_name("interpolated")

    apply_sub(scope, [y1_name, y0_name], delta_y_name, container, broadcast=0)
    apply_sub(scope, [x1_name, x0_name], delta_x_name, container, broadcast=0)
    apply_sub(scope, [reshaped_df_name, x0_name], offset_name, container, broadcast=0)
    apply_mul(scope, [delta_y_name, offset_name], scaled_name, container, broadcast=0)
    apply_div(scope, [scaled_name, delta_x_name], ratio_name, container, broadcast=0)
    apply_add(scope, [y0_name, ratio_name], interpolated_name, container, broadcast=0)

    if proto_type2 == TensorProto.DOUBLE:
        return interpolated_name

    cast_interpolated_name = scope.get_unique_variable_name("cast_interpolated")
    apply_cast(
        scope, interpolated_name, cast_interpolated_name, container, to=proto_type2
    )
    return cast_interpolated_name


def convert_calibrated_classifier_base_estimator(
    scope, operator, container, model, model_index
):
    # Temperature scaling uses _transform_temperature instead of per-class
    # calibration. The graph below describes sigmoid and isotonic calibration.
    # Computational graph:
    #
    # In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names
    # from the official ONNX spec:
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of instances
    # N: Number of features
    # C: Number of classes
    # CLASSIFIERCONVERTER: classifier converter corresponding to the op_type
    # a: slope in sigmoid model
    # b: intercept in sigmoid model
    # k: variable in the range [0, C)
    # input: input
    # class_prob_tensor: tensor with class probabilities(function output)
    #
    # Graph:
    #
    #   input [M, N] -> CLASSIFIERCONVERTER -> label [M]
    #                          |
    #                          V
    #                    probability_tensor [M, C]
    #                          |
    #         .----------------'---------.
    #         |                          |
    #         V                          V
    # ARRAYFEATUREEXTRACTOR <- k [1] -> ARRAYFEATUREEXTRACTOR
    #         |                          |
    #         V                          V
    #  transposed_df_col[M, 1] transposed_df_col[M, 1]
    #       |--------------------------|----------.--------------------------.
    #       |                          |          |                          |
    #       |if model.method='sigmoid' |          |if model.method='isotonic'|
    #       |                          |          |                          |
    #       V                          V          |linear interpolation      |
    #      MUL <-------- a -------->  MUL         V                          V
    #       |                          |          INTERP   ...           INTERP
    #       V                          V          |                          |
    #  a_df_prod [M, 1]  ... a_df_prod [M, 1]     V                          V
    #       |                          |  interp_df [M, 1] ... interp_df [M, 1]
    #       V                          V          |                          |
    #      ADD <--------- b ---------> ADD        '-------------------.------'
    #         |                          |                            |
    #         V                          V                            |
    #  exp_parameter [M, 1] ...   exp_parameter [M, 1]                |
    #         |                          |                            |
    #         V                          V                            |
    #        EXP        ...             EXP                           |
    #         |                          |                            |
    #         V                          V                            |
    #  exp_result [M, 1]  ...    exp_result [M, 1]                    |
    #         |                          |                            |
    #         V                          V                            |
    #       ADD <------- unity -------> ADD                           |
    #         |                          |                            |
    #         V                          V                            |
    #  denominator [M, 1]  ...   denominator [M, 1]                   |
    #         |                          |                            |
    #         V                          V                            |
    #        DIV <------- unity ------> DIV                           |
    #         |                          |                            |
    #         V                          V                            |
    # sigmoid_predict_result [M, 1] ... sigmoid_predict_result [M, 1] |
    #         |                          |                            |
    #         '-----.--------------------'                            |
    #               |-------------------------------------------------'
    #               |
    #               V
    #            CONCAT -> concatenated_prob [M, C]
    #                          |
    #        if  C = 2         |  if C != 2
    #      .-------------------'---------------------------.---------.
    #      |                                               |         |
    #      V                                               |         V
    # ARRAYFEATUREEXTRACTOR <- col_number [1]              |    REDUCESUM
    #                   |                                  |         |
    #                   '--------------------------------. |         |
    # unit_float_tensor [1] -> SUB <- first_col [M, 1] <-' |         |
    #                           |                         /          |
    #                           V                        V           V
    #                         CONCAT                    DIV <- reduced_prob [M]
    #                           |                        |
    #                           V                        |
    #                        class_prob_tensor [M, C] <--'
    model_proba = {RandomForestClassifier}

    if scope.get_options(operator.raw_operator, dict(nocl=False))["nocl"]:
        raise RuntimeError(
            "Option 'nocl' is not implemented for operator '{}'.".format(
                operator.raw_operator.__class__.__name__
            )
        )
    proto_type = guess_proto_type(operator.inputs[0].type)
    proto_type2 = proto_type
    if proto_type2 not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        proto_type2 = TensorProto.FLOAT
    base_model = (
        model.estimator if hasattr(model, "estimator") else model.base_estimator
    )
    while FrozenEstimator is not None and isinstance(base_model, FrozenEstimator):
        base_model = base_model.estimator
    op_type = sklearn_operator_name_map[type(base_model)]
    n_classes = (
        len(model.classes_) if hasattr(model, "classes_") else len(base_model.classes_)
    )
    prob_name = [None] * n_classes

    this_operator = scope.declare_local_operator(op_type, base_model)
    raw_scores = (
        container.has_options(base_model, "raw_scores")
        and type(base_model) not in model_proba
    )
    if model.method == "temperature":
        if container.target_opset < 9:
            raise RuntimeError("Temperature scaling requires opset >= 9.")
        raw_scores = hasattr(base_model, "decision_function")
        score_model = base_model
        while isinstance(score_model, Pipeline):
            score_model = score_model.steps[-1][1]
        if raw_scores and not container.has_options(score_model, "raw_scores"):
            raise NotImplementedError(
                "Temperature scaling requires decision_function export for %r."
                % type(score_model)
            )
    if raw_scores or (
        model.method == "temperature"
        and container.has_options(base_model, "raw_scores")
    ):
        container.add_options(id(base_model), {"raw_scores": raw_scores})
        scope.add_options(id(base_model), {"raw_scores": raw_scores})
    this_operator.inputs = operator.inputs
    label_name = scope.declare_local_variable("label", Int64TensorType())
    df_name = scope.declare_local_variable(
        "uncal_probability", operator.inputs[0].type.__class__()
    )
    this_operator.outputs.append(label_name)
    this_operator.outputs.append(df_name)
    df_inp = df_name.full_name

    if model.method == "temperature":
        # Match the fitted calibrator's score precision. sklearn 1.9 may fit
        # float32 linear models, whereas earlier versions use float64 scores.
        calibrators = (
            model.calibrators if hasattr(model, "calibrators") else model.calibrators_
        )
        calibration_type = TensorProto.DOUBLE
        if (
            proto_type2 != TensorProto.DOUBLE
            and np.asarray(calibrators[0].beta_).dtype == np.float32
        ):
            calibration_type = TensorProto.FLOAT
        scores_name = scope.get_unique_variable_name("temperature_scores")
        apply_cast(scope, df_inp, scores_name, container, to=calibration_type)
        probabilities_name = _transform_temperature(
            scope,
            container,
            model,
            scores_name,
            n_classes,
            raw_scores,
            calibration_type,
        )
        # sklearn accumulates each calibrated classifier's probabilities in
        # float64, and chooses labels before casting the public ONNX output.
        if calibration_type != TensorProto.DOUBLE:
            double_name = scope.get_unique_variable_name("temperature_double_proba")
            apply_cast(
                scope, probabilities_name, double_name, container, to=TensorProto.DOUBLE
            )
            return double_name
        return probabilities_name

    for k in range(n_classes):
        cur_k = k
        if n_classes == 2:
            cur_k += 1
        k_name = scope.get_unique_variable_name("k")
        df_col_name = scope.get_unique_variable_name(
            "tdf_col_%d_c%d" % (model_index, k)
        )
        prob_name[k] = scope.get_unique_variable_name(
            "prob_{}_c{}".format(model_index, k)
        )

        container.add_initializer(k_name, TensorProto.INT64, [], [cur_k])

        container.add_node(
            "ArrayFeatureExtractor",
            [df_inp, k_name],
            df_col_name,
            name=scope.get_unique_operator_name("CaliAFE_%d_c%d" % (model_index, k)),
            op_domain="ai.onnx.ml",
        )
        if model.method == "sigmoid":
            T = _transform_sigmoid(scope, container, model, df_col_name, k, proto_type)
        elif model.method == "isotonic":
            T = _transform_isotonic(scope, container, model, df_col_name, k, proto_type)
        else:
            raise ValueError("Unsupported calibration method %r." % model.method)

        prob_name[k] = T
        if n_classes == 2:
            break

    if n_classes == 2:
        zeroth_col_name = scope.get_unique_variable_name("zeroth_col%d" % model_index)
        merged_prob_name = scope.get_unique_variable_name("merged_prob%d" % model_index)
        unit_float_tensor_name = scope.get_unique_variable_name(
            "unit_float_tensor%d" % model_index
        )

        container.add_initializer(unit_float_tensor_name, proto_type2, [], [1.0])

        apply_sub(
            scope,
            [unit_float_tensor_name, prob_name[0]],
            zeroth_col_name,
            container,
            broadcast=1,
        )
        apply_concat(
            scope,
            [zeroth_col_name, prob_name[0]],
            merged_prob_name,
            container,
            axis=1,
            operator_name=scope.get_unique_variable_name("CaliConc%d" % model_index),
        )
        class_prob_tensor_name = merged_prob_name
    else:
        concatenated_prob_name = scope.get_unique_variable_name("concatenated_prob")
        reduced_prob_name = scope.get_unique_variable_name("reduced_prob")
        calc_prob_name = scope.get_unique_variable_name("calc_prob")

        apply_concat(scope, prob_name, concatenated_prob_name, container, axis=1)
        if container.target_opset < 13:
            container.add_node(
                "ReduceSum",
                concatenated_prob_name,
                reduced_prob_name,
                axes=[1],
                name=scope.get_unique_operator_name("ReduceSum"),
            )
        else:
            axis_name = scope.get_unique_variable_name("axis")
            container.add_initializer(axis_name, TensorProto.INT64, [1], [1])
            container.add_node(
                "ReduceSum",
                [concatenated_prob_name, axis_name],
                reduced_prob_name,
                name=scope.get_unique_operator_name("ReduceSum"),
            )
        num, deno = _handle_zeros(
            scope,
            container,
            concatenated_prob_name,
            reduced_prob_name,
            n_classes,
            proto_type,
        )
        apply_div(
            scope,
            [num, deno],
            calc_prob_name,
            container,
            broadcast=1,
            operator_name=scope.get_unique_variable_name("CaliDiv%d" % model_index),
        )
        class_prob_tensor_name = calc_prob_name
    return class_prob_tensor_name


def convert_sklearn_calibrated_classifier_cv(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    # Computational graph:
    #
    # In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names
    # from the official ONNX spec:
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of instances
    # N: Number of features
    # C: Number of classes
    # CONVERT_BASE_ESTIMATOR: base estimator convert function defined above
    # clf_length: number of calibrated classifiers
    # input: input
    # output: output
    # class_prob: class probabilities
    #
    # Graph:
    #
    #                         input [M, N]
    #                               |
    #           .-------------------|--------------------------.
    #           |                   |                          |
    #           V                   V                          V
    # CONVERT_BASE_ESTIMATOR  CONVERT_BASE_ESTIMATOR ... CONVERT_BASE_ESTIMATOR
    #           |                   |                          |
    #           V                   V                          V
    #  prob_scores_0 [M, C] prob_scores_1 [M, C] ... prob_scores_(clf_length-1)
    #           |                   |                          |  [M, C]
    #           '-------------------|--------------------------'
    #                               V
    #       add_result [M, C] <--- SUM
    #           |
    #           '--> DIV <- clf_length [1]
    #                 |
    #                 V
    #            class_prob [M, C] -> ARGMAX -> argmax_output [M, 1]
    #                                                   |
    #             classes -> ARRAYFEATUREEXTRACTOR  <---'
    #                               |
    #                               V
    #                            output [1]

    op = operator.raw_operator
    classes = op.classes_
    output_shape = (-1,)
    class_type = TensorProto.STRING
    proto_type = guess_proto_type(operator.inputs[0].type)
    proto_type2 = proto_type
    if proto_type2 not in (TensorProto.FLOAT, TensorProto.DOUBLE):
        proto_type2 = TensorProto.FLOAT

    if np.issubdtype(op.classes_.dtype, np.floating):
        class_type = TensorProto.INT32
        classes = classes.astype(np.int32)
    elif (
        np.issubdtype(op.classes_.dtype, np.signedinteger)
        or op.classes_.dtype == np.bool_
    ):
        class_type = TensorProto.INT32
    else:
        classes = np.array([s.encode("utf-8") for s in classes])

    clf_length = len(op.calibrated_classifiers_)
    prob_scores_name = []

    clf_length_name = scope.get_unique_variable_name("clf_length")
    classes_name = scope.get_unique_variable_name("classes")
    reshaped_result_name = scope.get_unique_variable_name("reshaped_result")
    argmax_output_name = scope.get_unique_variable_name("argmax_output")
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        "array_feature_extractor_result"
    )
    add_result_name = scope.get_unique_variable_name("add_result")

    container.add_initializer(classes_name, class_type, classes.shape, classes)
    calibration_type = TensorProto.DOUBLE if op.method == "temperature" else proto_type2
    container.add_initializer(clf_length_name, calibration_type, [], [clf_length])

    for clf_index, clf in enumerate(op.calibrated_classifiers_):
        prob_scores_name.append(
            convert_calibrated_classifier_base_estimator(
                scope, operator, container, clf, clf_index
            )
        )

    container.add_node(
        "Sum",
        list(prob_scores_name),
        add_result_name,
        op_version=7,
        name=scope.get_unique_operator_name("Sum"),
    )
    class_prob_name = operator.outputs[1].full_name
    if calibration_type != proto_type2:
        class_prob_name = scope.get_unique_variable_name("calibrated_probabilities")
    apply_div(
        scope,
        [add_result_name, clf_length_name],
        class_prob_name,
        container,
        broadcast=1,
    )
    if calibration_type != proto_type2:
        apply_cast(
            scope,
            class_prob_name,
            operator.outputs[1].full_name,
            container,
            to=proto_type2,
        )
    container.add_node(
        "ArgMax",
        class_prob_name,
        argmax_output_name,
        name=scope.get_unique_operator_name("ArgMax"),
        axis=1,
    )
    container.add_node(
        "ArrayFeatureExtractor",
        [classes_name, argmax_output_name],
        array_feature_extractor_result_name,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
    )

    if class_type == TensorProto.INT32:
        apply_reshape(
            scope,
            array_feature_extractor_result_name,
            reshaped_result_name,
            container,
            desired_shape=output_shape,
        )
        apply_cast(
            scope,
            reshaped_result_name,
            operator.outputs[0].full_name,
            container,
            to=TensorProto.INT64,
        )
    else:
        apply_reshape(
            scope,
            array_feature_extractor_result_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=output_shape,
        )


register_converter(
    "SklearnCalibratedClassifierCV",
    convert_sklearn_calibrated_classifier_cv,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
