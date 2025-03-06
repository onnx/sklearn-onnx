# SPDX-License-Identifier: Apache-2.0


import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_tensor,
    tensor_dtype_to_np_dtype,
    make_tensor_value_info,
    make_graph,
    make_node,
)
from onnx.numpy_helper import from_array
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxArgMax,
    OnnxArrayFeatureExtractor,
    OnnxCast,
    OnnxConcat,
    OnnxDiv,
    OnnxEqual,
    OnnxFlatten,
    OnnxIdentity,
    OnnxLess,
    OnnxMatMul,
    OnnxMax,
    OnnxMul,
    OnnxNeg,
    OnnxReciprocal,
    OnnxReduceMeanApi18,
    OnnxReduceSumApi11,
    OnnxReshapeApi13,
    OnnxShape,
    OnnxSqueezeApi11,
    OnnxSub,
    OnnxTopK_1,
    OnnxTranspose,
)

try:
    from ..algebra.onnx_ops import (
        OnnxConstantOfShape,
        OnnxCumSum,
        OnnxIsNaN,
        OnnxWhere,
    )
except ImportError:
    OnnxConstantOfShape = None
    OnnxCumSum = None
    OnnxIsNaN = None
    OnnxWhere = None
try:
    from ..algebra.onnx_ops import OnnxTopK_10
except ImportError:
    OnnxTopK_10 = None
try:
    from ..algebra.onnx_ops import OnnxTopK_11
except ImportError:
    OnnxTopK_11 = None
from ..algebra.complex_functions import onnx_cdist, _onnx_cdist_sqeuclidean
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import (
    Int64TensorType,
    DoubleTensorType,
    guess_numpy_type,
    guess_proto_type,
)
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto
from ._gp_kernels import py_make_float_array


def onnx_nearest_neighbors_indices_k(
    X,
    Y,
    k,
    metric="euclidean",
    dtype=None,
    op_version=None,
    keep_distances=False,
    optim=None,
    **kwargs,
):
    """
    Retrieves the nearest neigbours *ONNX*.
    :param X: features or *OnnxOperatorMixin*
    :param Y: neighbours or *OnnxOperatorMixin*
    :param k: number of neighbours to retrieve
    :param metric: requires metric
    :param dtype: numerical type
    :param op_version: opset version
    :param keep_distance: returns the distances as well (second position)
    :param optim: implements specific optimisations,
        ``'cdist'`` replaces *Scan* operator by operator *CDist*
    :param kwargs: additional parameters for function @see fn onnx_cdist
    :return: top indices, top distances
    """
    kwargs_dist = {k: v for k, v in kwargs.items() if k == "p"}
    kwargs_topk = {k: v for k, v in kwargs.items() if k != "p"}
    if optim == "cdist":
        from skl2onnx.algebra.custom_ops import OnnxCDist

        dist = OnnxCDist(X, Y, metric=metric, op_version=op_version, **kwargs_dist)
    elif optim is None:
        dim_in = Y.shape[1] if hasattr(Y, "shape") else None
        dim_out = Y.shape[0] if hasattr(Y, "shape") else None
        dist = onnx_cdist(
            X,
            Y,
            metric=metric,
            dtype=dtype,
            op_version=op_version,
            dim_in=dim_in,
            dim_out=dim_out,
            **kwargs_dist,
        )
    else:
        raise ValueError("Unknown optimisation '{}'.".format(optim))
    if op_version < 10:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_1(neg_dist, k=k, op_version=1, **kwargs_topk)
    elif op_version < 11:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_10(
            neg_dist, np.array([k], dtype=np.int64), op_version=10, **kwargs_topk
        )
    else:
        node = OnnxTopK_11(
            dist,
            np.array([k], dtype=np.int64),
            largest=0,
            sorted=1,
            op_version=11,
            **kwargs_topk,
        )
        if keep_distances:
            return (
                node[1],
                OnnxMul(node[0], np.array([-1], dtype=dtype), op_version=op_version),
            )
    if keep_distances:
        return (node[1], node[0])
    return node[1]


def onnx_nearest_neighbors_indices_radius(
    X,
    Y,
    radius,
    metric="euclidean",
    dtype=None,
    op_version=None,
    keep_distances=False,
    optim=None,
    proto_dtype=None,
    **kwargs,
):
    """
    Retrieves the nearest neigbours *ONNX*.
    :param X: features or *OnnxOperatorMixin*
    :param Y: neighbours or *OnnxOperatorMixin*
    :param radius: radius
    :param metric: requires metric
    :param dtype: numerical type
    :param op_version: opset version
    :param keep_distance: returns the distances as well (second position)
    :param optim: implements specific optimisations,
        ``'cdist'`` replaces *Scan* operator by operator *CDist*
    :param kwargs: additional parameters for function @see fn onnx_cdist
    :return: 3 squares matrices, indices or -1, distance or 0,
        based on the fact that the distance is below the radius,
        binary weights
    """
    opv = op_version
    if optim == "cdist":
        from skl2onnx.algebra.custom_ops import OnnxCDist

        dist = OnnxCDist(X, Y, metric=metric, op_version=op_version, **kwargs)
    elif optim is None:
        dim_in = Y.shape[1] if hasattr(Y, "shape") else None
        dim_out = Y.shape[0] if hasattr(Y, "shape") else None
        dist = onnx_cdist(
            X,
            Y,
            metric=metric,
            dtype=dtype,
            op_version=op_version,
            dim_in=dim_in,
            dim_out=dim_out,
            **kwargs,
        )
    else:
        raise ValueError("Unknown optimisation '{}'.".format(optim))

    less = OnnxLess(dist, np.array([radius], dtype=dtype), op_version=opv)
    less.set_onnx_name_prefix("cond")
    shape = OnnxShape(dist, op_version=opv)
    zero = OnnxCast(
        OnnxConstantOfShape(shape, op_version=opv), op_version=opv, to=proto_dtype
    )
    tensor_value = py_make_float_array(-1, dtype=dtype, as_tensor=True)
    minus = OnnxCast(
        OnnxConstantOfShape(shape, op_version=opv, value=tensor_value),
        op_version=opv,
        to=onnx_proto.TensorProto.INT64,
    )
    minus_range = OnnxAdd(
        OnnxNeg(
            OnnxCumSum(minus, np.array([1], dtype=np.int64), op_version=opv),
            op_version=opv,
        ),
        minus,
        op_version=opv,
    )
    minus_range.set_onnx_name_prefix("arange")

    dist_only = OnnxWhere(less, dist, zero, op_version=opv)
    dist_only.set_onnx_name_prefix("nndist")
    indices = OnnxWhere(less, minus_range, minus, op_version=opv)
    indices.set_onnx_name_prefix("nnind")
    binary = OnnxCast(less, to=proto_dtype, op_version=opv)
    binary.set_onnx_name_prefix("nnbin")
    return indices, dist_only, binary


def _convert_nearest_neighbors(operator, container, k=None, radius=None):
    """
    Common parts to regressor and classifier. Let's denote
    *N* as the number of observations, *k*
    the number of neighbours. It returns
    the following intermediate results:

    top_indices: [N, k] (int64), best indices for
        every observation
    top_distances: [N, k] (dtype), float distances
        for every observation, it can be None
        if the weights are uniform
    top_labels: [N, k] (label type), labels
        associated to every top index
    weights: [N, k] (dtype), if top_distances is not None,
        returns weights
    norm: [N] (dtype), if top_distances is not None,
        returns normalized weights
    axis: 1 if there is one dimension only, 2 if
        this is a multi-regression or a multi classification
    """
    X = operator.inputs[0]
    op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    proto_type = guess_proto_type(X.type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT

    if isinstance(X.type, Int64TensorType):
        X = OnnxCast(X, to=proto_type, op_version=opv)

    options = container.get_options(op, dict(optim=None))

    single_reg = (
        not hasattr(op, "_y")
        or len(op._y.shape) == 1
        or (len(op._y.shape) == 2 and op._y.shape[1] == 1)
    )
    ndim = 1 if single_reg else op._y.shape[1]

    metric = op.effective_metric_ if hasattr(op, "effective_metric_") else op.metric
    neighb = op._fit_X.astype(dtype)

    if (
        hasattr(op, "n_neighbors")
        and op.n_neighbors is not None
        and hasattr(op, "radius")
        and op.radius is not None
    ):
        raise RuntimeError(
            "The model defines radius and n_neighbors at the "
            "same time ({} and {}). "
            "This case is not supported.".format(op.radius, op.n_neighbors)
        )

    if hasattr(op, "n_neighbors") and op.n_neighbors is not None:
        k = op.n_neighbors if k is None else k
        radius = None
    elif hasattr(op, "radius") and op.radius is not None:
        k = None
        radius = op.radius if radius is None else radius
    else:
        raise RuntimeError("Cannot convert class '{}'.".format(op.__class__.__name__))

    training_labels = op._y if hasattr(op, "_y") else None
    distance_kwargs = {}
    if metric == "minkowski":
        if op.p != 2:
            distance_kwargs["p"] = op.p
        else:
            metric = "euclidean"

    weights = op.weights if hasattr(op, "weights") else "distance"
    binary = None
    if weights == "uniform" and radius is None:
        top_indices = onnx_nearest_neighbors_indices_k(
            X,
            neighb,
            k,
            metric=metric,
            dtype=dtype,
            op_version=opv,
            optim=options.get("optim", None),
            **distance_kwargs,
        )
        top_distances = None
    elif radius is not None:
        three = onnx_nearest_neighbors_indices_radius(
            X,
            neighb,
            radius,
            metric=metric,
            dtype=dtype,
            op_version=opv,
            keep_distances=True,
            proto_dtype=proto_type,
            optim=options.get("optim", None),
            **distance_kwargs,
        )
        top_indices, top_distances, binary = three
    elif weights == "distance":
        top_indices, top_distances = onnx_nearest_neighbors_indices_k(
            X,
            neighb,
            k,
            metric=metric,
            dtype=dtype,
            op_version=opv,
            keep_distances=True,
            optim=options.get("optim", None),
            **distance_kwargs,
        )
    else:
        raise RuntimeError(
            "Unable to convert KNeighborsRegressor when weights is callable."
        )

    if training_labels is not None:
        if ndim > 1:
            training_labels = training_labels.T
            axis = 2
        else:
            training_labels = training_labels.ravel()
            axis = 1
        if opv >= 9:
            kor = k if k is not None else training_labels.shape[-1]
            if ndim > 1:
                shape = np.array([ndim, -1, kor], dtype=np.int64)
            else:
                shape = np.array([-1, kor], dtype=np.int64)
        else:
            raise RuntimeError(
                "Conversion of a KNeighborsRegressor for multi regression "
                "requires opset >= 9."
            )

        if training_labels.dtype == np.int32:
            training_labels = training_labels.astype(np.int64)
        flattened = OnnxFlatten(top_indices, op_version=opv)
        extracted = OnnxArrayFeatureExtractor(
            training_labels, flattened, op_version=opv
        )
        reshaped = OnnxReshapeApi13(extracted, shape, op_version=opv)

        if ndim > 1:
            reshaped = OnnxTranspose(reshaped, op_version=opv, perm=[1, 0, 2])
        reshaped.set_onnx_name_prefix("knny")

    else:
        reshaped = None
        axis = 1

    if binary is not None:
        if op.weights == "uniform":
            wei = binary
        else:
            modified = OnnxMax(
                top_distances, np.array([1e-6], dtype=dtype), op_version=opv
            )
            wei = OnnxMul(
                binary, OnnxReciprocal(modified, op_version=opv), op_version=opv
            )
        norm = OnnxReduceSumApi11(wei, op_version=opv, axes=[1], keepdims=0)
    elif top_distances is not None:
        modified = OnnxMax(top_distances, np.array([1e-6], dtype=dtype), op_version=opv)
        wei = OnnxReciprocal(modified, op_version=opv)
        norm = OnnxReduceSumApi11(wei, op_version=opv, axes=[1], keepdims=0)
    else:
        norm = None
        wei = None

    if wei is not None:
        wei.set_onnx_name_prefix("wei")
    if norm is not None:
        norm.set_onnx_name_prefix("norm")
    return top_indices, top_distances, reshaped, wei, norm, axis


def convert_nearest_neighbors_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNeighborsRegressor* into *ONNX*.
    The converted model may return different predictions depending
    on how the runtime select the topk element.
    *scikit-learn* uses function `argpartition
    <https://docs.scipy.org/doc/numpy/reference/
    generated/numpy.argpartition.html>`_ which keeps the
    original order of the elements.
    """
    many = _convert_nearest_neighbors(operator, container)
    _, top_distances, reshaped, wei, norm, axis = many
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT

    opv = container.target_opset
    out = operator.outputs

    reshaped_cast = OnnxCast(reshaped, to=proto_type, op_version=opv)
    if top_distances is not None:
        # Multi-target
        if (
            hasattr(operator.raw_operator, "_y")
            and len(operator.raw_operator._y.shape) > 1
            and operator.raw_operator._y.shape[1] > 1
        ):
            rs = OnnxTranspose(reshaped_cast, perm=[1, 0, 2], op_version=opv)
            weighted_rs = OnnxMul(rs, wei, op_version=opv)
            weighted = OnnxTranspose(weighted_rs, perm=[1, 0, 2], op_version=opv)

            if OnnxIsNaN is not None:
                # This steps sometimes produces nan (bug in onnxuntime)
                # They are replaced by null values.
                isnan = OnnxIsNaN(weighted, op_version=opv)
                shape = OnnxShape(weighted, op_version=opv)
                csts0 = OnnxConstantOfShape(shape, op_version=opv)
                weighted = OnnxWhere(isnan, csts0, weighted, op_version=opv)
                # Back to original plan.

            res = OnnxReduceSumApi11(weighted, axes=[axis], op_version=opv, keepdims=0)
            norm2 = OnnxReshapeApi13(
                norm, np.array([-1, 1], dtype=np.int64), op_version=opv
            )
            res = OnnxDiv(res, norm2, op_version=opv, output_names=out)
        else:
            weighted = OnnxMul(reshaped_cast, wei, op_version=opv)
            res = OnnxReduceSumApi11(weighted, axes=[axis], op_version=opv, keepdims=0)
            res.set_onnx_name_prefix("final")
            if opv >= 12:
                shape = OnnxShape(res, op_version=opv)
                norm = OnnxReshapeApi13(norm, shape, op_version=opv)
                norm.set_onnx_name_prefix("normr")
            res = OnnxDiv(res, norm, op_version=opv)
            res = OnnxReshapeApi13(
                res, np.array([-1, 1], dtype=np.int64), output_names=out, op_version=opv
            )
    else:
        if (
            hasattr(operator.raw_operator, "_y")
            and len(np.squeeze(operator.raw_operator._y).shape) == 1
        ):
            keepdims = 1
        elif operator.raw_operator.n_neighbors == 1:
            keepdims = 0
        else:
            keepdims = 0
        res = OnnxReduceMeanApi18(
            reshaped_cast,
            axes=[axis],
            op_version=opv,
            keepdims=keepdims,
            output_names=out,
        )
    res.add_to(scope, container)


def get_proba_and_label(
    container, nb_classes, reshaped, wei, axis, opv, proto_type, keep_axis=True
):
    """
    This function calculates the label by choosing majority label
    amongst the nearest neighbours.
    """
    conc = []
    for cl in range(nb_classes):
        cst = np.array([cl], dtype=np.int64)
        mat_cast = OnnxCast(
            OnnxEqual(reshaped, cst, op_version=opv), op_version=opv, to=proto_type
        )
        if wei is not None:
            if not keep_axis:
                mat_cast = OnnxSqueezeApi11(mat_cast, axes=[-1], op_version=opv)
            mat_cast = OnnxMul(mat_cast, wei, op_version=opv)
        wh = OnnxReduceSumApi11(mat_cast, axes=[1], op_version=opv)
        conc.append(wh)
    all_together = OnnxConcat(*conc, axis=1, op_version=opv)
    sum_prob = OnnxReduceSumApi11(all_together, axes=[1], op_version=opv, keepdims=1)
    res = OnnxArgMax(all_together, axis=axis, op_version=opv, keepdims=0)
    return all_together, sum_prob, res


def convert_nearest_neighbors_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNeighborsClassifier* into *ONNX*.
    The converted model may return different predictions depending
    on how the runtime select the topk element.
    *scikit-learn* uses function `argpartition
    <https://docs.scipy.org/doc/numpy/reference/
    generated/numpy.argpartition.html>`_ which keeps the
    original order of the elements.
    """
    many = _convert_nearest_neighbors(operator, container)
    _, __, reshaped, wei, ___, axis = many

    opv = container.target_opset
    out = operator.outputs
    op = operator.raw_operator
    nb_classes = len(op.classes_)
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT

    if axis == 0:
        raise RuntimeError(
            "Binary classification not implemented in scikit-learn. "
            "Check this code is not reused for other libraries."
        )
    classes = get_label_classes(scope, op)
    if hasattr(classes, "dtype") and (
        np.issubdtype(classes.dtype, np.floating) or classes.dtype == np.bool_
    ):
        classes = classes.astype(np.int32)
        is_integer = True
    elif isinstance(classes[0], (int, np.int32, np.int64)):
        is_integer = True
    else:
        is_integer = False
    if isinstance(op.classes_, list) and isinstance(op.classes_[0], np.ndarray):
        # Multi-label
        out_labels, out_probas = [], []
        for index, cur_class in enumerate(op.classes_):
            transpose_result = OnnxTranspose(reshaped, op_version=opv, perm=[0, 2, 1])
            extracted_name = OnnxArrayFeatureExtractor(
                transpose_result, np.array([index], dtype=np.int64), op_version=opv
            )
            extracted_name.set_onnx_name_prefix("tr%d" % index)
            all_together, sum_prob, res = get_proba_and_label(
                container,
                len(cur_class),
                extracted_name,
                wei,
                1,
                opv,
                proto_type,
                keep_axis=False,
            )
            probas = OnnxDiv(all_together, sum_prob, op_version=opv)
            res_name = OnnxArrayFeatureExtractor(cur_class, res, op_version=opv)
            res_name.set_onnx_name_prefix("div%d" % index)
            reshaped_labels = OnnxReshapeApi13(
                res_name, np.array([-1, 1], dtype=np.int64), op_version=opv
            )
            reshaped_probas = OnnxReshapeApi13(
                probas,
                np.array([1, -1, len(cur_class)], dtype=np.int64),
                op_version=opv,
            )
            out_labels.append(reshaped_labels)
            out_probas.append(reshaped_probas)
        concatenated_labels = OnnxConcat(*out_labels, axis=1, op_version=opv)
        final_proba = OnnxConcat(
            *out_probas, axis=0, output_names=out[1:], op_version=opv
        )
        final_label = OnnxCast(
            concatenated_labels,
            to=onnx_proto.TensorProto.INT64,
            output_names=out[:1],
            op_version=opv,
        )
        final_label.add_to(scope, container)
        final_proba.add_to(scope, container)
    else:
        all_together, sum_prob, res = get_proba_and_label(
            container, nb_classes, reshaped, wei, axis, opv, proto_type
        )
        probas = OnnxDiv(all_together, sum_prob, op_version=opv, output_names=out[1:])
        probas.set_onnx_name_prefix("bprob")
        res_name = OnnxArrayFeatureExtractor(classes, res, op_version=opv)
        if is_integer:
            res_name = OnnxCast(
                res_name, to=onnx_proto.TensorProto.INT64, op_version=opv
            )
        out_labels = OnnxReshapeApi13(
            res_name,
            np.array([-1], dtype=np.int64),
            output_names=out[:1],
            op_version=opv,
        )
        out_labels.set_onnx_name_prefix("blab")
        out_labels.add_to(scope, container)
        probas.add_to(scope, container)


def convert_nearest_neighbors_transform(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *NearestNeighbors* into *ONNX*.
    """
    many = _convert_nearest_neighbors(operator, container)
    top_indices, top_distances = many[:2]
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32

    out = operator.outputs

    ind = OnnxIdentity(
        top_indices, output_names=out[:1], op_version=container.target_opset
    )
    dist = OnnxMul(
        top_distances,
        np.array([-1], dtype=dtype),
        output_names=out[1:],
        op_version=container.target_opset,
    )

    dist.add_to(scope, container)
    ind.add_to(scope, container)


def convert_k_neighbours_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNeighborsTransformer* into *ONNX*.
    """
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    transformer_op = operator.raw_operator
    op_version = container.target_opset
    k = (
        transformer_op.n_neighbors + 1
        if transformer_op.mode == "distance"
        else transformer_op.n_neighbors
    )
    out = operator.outputs

    many = _convert_nearest_neighbors(operator, container, k=k)
    top_indices, top_dist = many[:2]
    top_dist = (
        OnnxReshapeApi13(
            OnnxMul(top_dist, np.array([-1], dtype=dtype), op_version=op_version),
            np.array([-1, 1, k], dtype=np.int64),
            op_version=op_version,
        )
        if transformer_op.mode == "distance"
        else None
    )
    fit_samples_indices = np.array(
        np.arange(transformer_op.n_samples_fit_).reshape((1, -1, 1)), dtype=np.int64
    )
    reshaped_ind = OnnxReshapeApi13(
        top_indices, np.array([-1, 1, k], dtype=np.int64), op_version=op_version
    )
    comparison_res = OnnxCast(
        OnnxEqual(fit_samples_indices, reshaped_ind, op_version=op_version),
        op_version=op_version,
        to=proto_type,
    )
    if top_dist:
        comparison_res = OnnxMul(comparison_res, top_dist, op_version=op_version)
    res = OnnxReduceSumApi11(
        comparison_res,
        op_version=op_version,
        axes=[2],
        keepdims=0,
        output_names=out[:1],
    )
    res.add_to(scope, container)


def _nan_euclidean_distance(
    container, model, input_name, op_version, optim, dtype, proto_type
):
    training_data = model._fit_X.astype(dtype)
    shape = OnnxShape(input_name, op_version=op_version)
    zero = OnnxConstantOfShape(
        shape, value=make_tensor("value", proto_type, (1,), [0]), op_version=op_version
    )
    missing_input_name = OnnxIsNaN(input_name, op_version=op_version)
    masked_input_name = OnnxWhere(
        missing_input_name, zero, input_name, op_version=op_version
    )
    missing_y = np.isnan(training_data)
    training_data[missing_y] = 0
    d_in = training_data.shape[1] if hasattr(training_data, "shape") else None
    d_out = training_data.shape[0] if hasattr(training_data, "shape") else None

    if optim is None:
        dist = _onnx_cdist_sqeuclidean(
            masked_input_name,
            training_data,
            dtype=dtype,
            op_version=container.target_opset,
            dim_in=d_in,
            dim_out=d_out,
        )
    elif optim == "cdist":
        from skl2onnx.algebra.custom_ops import OnnxCDist

        dist = OnnxCDist(
            masked_input_name,
            training_data,
            metric="sqeuclidean",
            op_version=container.target_opset,
        )
    else:
        raise RuntimeError("Unexpected optimization '{}'.".format(optim))
    dist1 = OnnxMatMul(
        OnnxMul(masked_input_name, masked_input_name, op_version=op_version),
        missing_y.T.astype(dtype),
        op_version=op_version,
    )
    dist2 = OnnxMatMul(
        OnnxCast(missing_input_name, to=proto_type, op_version=op_version),
        (training_data * training_data).T.astype(dtype),
        op_version=op_version,
    )
    distances = OnnxSub(
        dist, OnnxAdd(dist1, dist2, op_version=op_version), op_version=op_version
    )
    present_x = OnnxSub(
        np.array([1], dtype=dtype),
        OnnxCast(missing_input_name, to=proto_type, op_version=op_version),
        op_version=op_version,
    )
    present_y = (1.0 - missing_y).astype(dtype)
    present_count = OnnxMatMul(
        present_x, present_y.T.astype(dtype), op_version=op_version
    )
    present_count = OnnxMax(
        np.array([1], dtype=dtype), present_count, op_version=op_version
    )
    dist = OnnxDiv(distances, present_count, op_version=op_version)
    return (
        OnnxMul(dist, np.array([d_in], dtype=dtype), op_version=op_version),
        missing_input_name,
    )


def _nearest_neighbours(
    container, model, input_name, op_version, optim, dtype, proto_type, **kwargs
):
    dist, missing_input_name = _nan_euclidean_distance(
        container, model, input_name, op_version, optim, dtype, proto_type
    )
    if op_version < 10:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_1(neg_dist, k=model.n_neighbors, op_version=1, **kwargs)
    elif op_version < 11:
        neg_dist = OnnxMul(dist, np.array([-1], dtype=dtype), op_version=op_version)
        node = OnnxTopK_10(
            neg_dist,
            np.array([model.n_neighbors], dtype=np.int64),
            op_version=10,
            **kwargs,
        )
    else:
        node = OnnxTopK_11(
            dist,
            np.array([model.n_neighbors], dtype=np.int64),
            largest=0,
            sorted=1,
            op_version=11,
            **kwargs,
        )
    return node[1], missing_input_name


def make_dict_idx_map(g: ModelComponentContainer, scope: Scope, itype: int):
    gr = ModelComponentContainer({"": g.main_opset}, as_function=True)
    x = gr.make_tensor_input("x")
    row_missing_idx = gr.make_tensor_input("row_missing_idx")
    op = gr.get_op_builder(scope)
    init7_s_0 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value"),
        outputs=["init7_s_0"],
    )
    init7_s_1 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value"),
        outputs=["init7_s_1"],
    )
    init7_s1__1 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value"),
        outputs=["init7_s1__1"],
    )
    _shape_x0 = op.Shape(x, end=1, start=0, outputs=["_shape_x0"])
    _shape_row_missing_idx0 = op.Shape(
        row_missing_idx, end=1, start=0, outputs=["_shape_row_missing_idx0"]
    )
    sym_size_int_4 = op.SqueezeAnyOpset(
        _shape_row_missing_idx0, outputs=["sym_size_int_4"]
    )
    zeros = op.ConstantOfShape(
        _shape_x0,
        value=from_array(np.array([0], dtype=np.int64), name="value"),
        outputs=["zeros"],
    )
    arange = op.Range(init7_s_0, sym_size_int_4, init7_s_1, outputs=["arange"])
    _onx_unsqueeze_row_missing_idx0 = op.UnsqueezeAnyOpset(
        row_missing_idx, init7_s1__1, outputs=["_onx_unsqueeze_row_missing_idx0"]
    )
    output_0 = op.ScatterND(
        zeros, _onx_unsqueeze_row_missing_idx0, arange, outputs=["output_0"]
    )
    gr.make_tensor_output(output_0)
    g.make_local_function(
        container=gr, optimize=False, name="dict_idx_map", domain="local_domain"
    )
    return gr


def make_dist_nan_euclidean(g: ModelComponentContainer, scope: Scope, itype: int):
    gr = ModelComponentContainer({"": g.main_opset}, as_function=True)
    x = gr.make_tensor_input("x")
    y = gr.make_tensor_input("y")
    op = gr.get_op_builder(scope)
    c_lifted_tensor_0 = op.Constant(
        value=from_array(
            np.array(0.0, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["c_lifted_tensor_0"],
    )
    c_lifted_tensor_1 = op.Constant(
        value=from_array(
            np.array(0.0, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["c_lifted_tensor_1"],
    )
    c_lifted_tensor_2 = op.Constant(
        value=from_array(
            np.array(np.nan, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["c_lifted_tensor_2"],
    )
    c_lifted_tensor_3 = op.Constant(
        value=from_array(
            np.array([1.0], dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["c_lifted_tensor_3"],
    )
    init1_s_ = op.Constant(
        value=from_array(
            np.array(-2.0, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["init1_s_"],
    )
    init7_s1_1 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value"),
        outputs=["init7_s1_1"],
    )
    init1_s1_ = op.Constant(
        value=from_array(
            np.array([0.0], dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["init1_s1_"],
    )
    init1_s_2 = op.Constant(
        value=from_array(
            np.array(1.0, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["init1_s_2"],
    )
    init1_s_3 = op.Constant(
        value=from_array(
            np.array(0.0, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["init1_s_3"],
    )
    init7_s2_1__1 = op.Constant(
        value=from_array(np.array([1, -1], dtype=np.int64), name="value"),
        outputs=["init7_s2_1__1"],
    )
    isnan = op.IsNaN(x, outputs=["isnan"])
    _to_copy_2 = op.Cast(isnan, to=itype, outputs=["_to_copy_2"])
    isnan_1 = op.IsNaN(y, outputs=["isnan_1"])
    index_put = op.Where(isnan, c_lifted_tensor_0, x, outputs=["index_put"])
    index_put_1 = op.Where(isnan_1, c_lifted_tensor_1, y, outputs=["index_put_1"])
    mul_18 = op.Mul(index_put, index_put, outputs=["mul_18"])
    mul_21 = op.Mul(index_put_1, index_put_1, outputs=["mul_21"])
    matmul_2 = op.Gemm(_to_copy_2, mul_21, transA=0, transB=1, outputs=["matmul_2"])
    _reshape_init1_s_0 = op.Reshape(
        init1_s_, init7_s1_1, outputs=["_reshape_init1_s_0"]
    )
    index_put__2 = op.Mul(index_put, _reshape_init1_s_0, outputs=["index_put__2"])
    matmul = op.Gemm(index_put__2, index_put_1, transA=0, transB=1, outputs=["matmul"])
    sum_1 = op.ReduceSumAnyOpset(mul_18, init7_s1_1, keepdims=1, outputs=["sum_1"])
    add_50 = op.Add(matmul, sum_1, outputs=["add_50"])
    sum_2 = op.ReduceSumAnyOpset(mul_21, init7_s1_1, keepdims=1, outputs=["sum_2"])
    permute_2 = op.Reshape(sum_2, init7_s2_1__1, outputs=["permute_2"])
    add_59 = op.Add(add_50, permute_2, outputs=["add_59"])
    _to_copy_1 = op.Cast(isnan_1, to=itype, outputs=["_to_copy_1"])
    matmul_1 = op.Gemm(mul_18, _to_copy_1, transA=0, transB=1, outputs=["matmul_1"])
    sub_32 = op.Sub(add_59, matmul_1, outputs=["sub_32"])
    sub_43 = op.Sub(sub_32, matmul_2, outputs=["sub_43"])
    clip = op.Clip(sub_43, init1_s1_, outputs=["clip"])
    _reshape_init1_s_20 = op.Reshape(
        init1_s_2, init7_s1_1, outputs=["_reshape_init1_s_20"]
    )
    rsub = op.Sub(_reshape_init1_s_20, _to_copy_2, outputs=["rsub"])
    bitwise_not = op.Not(isnan_1, outputs=["bitwise_not"])
    _to_copy_4 = op.Cast(bitwise_not, to=itype, outputs=["_to_copy_4"])
    matmul_3 = op.Gemm(rsub, _to_copy_4, transA=0, transB=1, outputs=["matmul_3"])
    _reshape_init1_s_30 = op.Reshape(
        init1_s_3, init7_s1_1, outputs=["_reshape_init1_s_30"]
    )
    eq_61 = op.Equal(matmul_3, _reshape_init1_s_30, outputs=["eq_61"])
    index_put_2 = op.Where(eq_61, c_lifted_tensor_2, clip, outputs=["index_put_2"])
    maximum = op.Max(c_lifted_tensor_3, matmul_3, outputs=["maximum"])
    div = op.Div(index_put_2, maximum, outputs=["div"])
    col_size = op.Shape(x, start=1, end=2, outputs=["col_size"])
    col_size_c = op.Cast(col_size, to=itype)
    _onx_mul_div0 = op.Mul(div, col_size_c, outputs=["_onx_mul_div0"])
    output_0 = op.Sqrt(_onx_mul_div0, outputs=["output_0"])
    gr.make_tensor_output(output_0)
    g.make_local_function(
        container=gr, optimize=False, name="dist_nan_euclidean", domain="local_domain"
    )
    return gr


def make_calc_impute_donors(g: ModelComponentContainer, scope: Scope, itype: int):
    gr = ModelComponentContainer({"": g.main_opset}, as_function=True)
    dist_pot_donors = gr.make_tensor_input("dist_pot_donors")
    n_neighbors = gr.make_tensor_input("n_neighbors")
    op = gr.get_op_builder(scope)
    init7_s_0 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value"),
        outputs=["init7_s_0"],
    )
    init7_s1_1 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value"),
        outputs=["init7_s1_1"],
    )
    init7_s_1 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value"),
        outputs=["init7_s_1"],
    )
    _shape_dist_pot_donors0 = op.Shape(
        dist_pot_donors, end=1, start=0, outputs=["_shape_dist_pot_donors0"]
    )
    sym_size_int_4 = op.SqueezeAnyOpset(
        _shape_dist_pot_donors0, outputs=["sym_size_int_4"]
    )
    if g.target_opset < 11:
        unused_topk_values, neg_output_0 = op.TopK(
            op.Neg(dist_pot_donors),
            op.Reshape(n_neighbors, np.array([1], dtype=np.int64)),
            largest=1,
            sorted=1,
            outputs=["unused_topk_values", "neg_output_0"],
        )
        output_0 = op.Neg(neg_output_0, outputs=["output_0"])
    else:
        max_value = (
            np.finfo(np.float32).max
            if itype == TensorProto.FLOAT
            else np.finfo(np.float64).max
        )
        init_max_value = op.Constant(
            value=from_array(
                np.array(max_value, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
            ),
            outputs=["init_max_value"],
        )

        unused_topk_values, output_0 = op.TopK(
            op.Where(op.IsNaN(dist_pot_donors), init_max_value, dist_pot_donors),
            op.Reshape(n_neighbors, np.array([1], dtype=np.int64)),
            largest=0,
            sorted=1,
            outputs=["unused_topk_values", "output_0"],
        )

    arange = op.Range(init7_s_0, sym_size_int_4, init7_s_1, outputs=["arange"])
    unsqueeze = op.UnsqueezeAnyOpset(arange, init7_s1_1, outputs=["unsqueeze"])
    _onx_gathernd_dist_pot_donors0 = op.GatherND(
        dist_pot_donors,
        unsqueeze,
        batch_dims=0,
        outputs=["_onx_gathernd_dist_pot_donors0"],
    )
    output_1 = op.GatherElements(
        _onx_gathernd_dist_pot_donors0, output_0, axis=1, outputs=["output_1"]
    )
    gr.make_tensor_output(output_0)
    gr.make_tensor_output(output_1)
    g.make_local_function(
        container=gr, optimize=False, name="calc_impute_donors", domain="local_domain"
    )
    return gr


def make_calc_impute_weights(g: ModelComponentContainer, scope: Scope, itype: int):
    gr = ModelComponentContainer({"": g.main_opset}, as_function=True)
    donors_dist = gr.make_tensor_input("donors_dist")
    op = gr.get_op_builder(scope)
    c_lifted_tensor_0 = op.Constant(
        value=from_array(
            np.array(0.0, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["c_lifted_tensor_0"],
    )
    _shape_donors_dist0 = op.Shape(donors_dist, outputs=["_shape_donors_dist0"])
    ones_like = op.ConstantOfShape(
        _shape_donors_dist0,
        value=from_array(
            np.array([1.0], dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["ones_like"],
    )
    isnan = op.IsNaN(donors_dist, outputs=["isnan"])
    output_0 = op.Where(isnan, c_lifted_tensor_0, ones_like, outputs=["output_0"])
    gr.make_tensor_output(output_0)
    g.make_local_function(
        container=gr, optimize=False, name="calc_impute_weights", domain="local_domain"
    )
    return gr


def make_calc_impute_make_new_neights(
    g: ModelComponentContainer, scope: Scope, itype: int
):
    gr = ModelComponentContainer({"": g.main_opset}, as_function=True)
    donors_mask = gr.make_tensor_input("donors_mask")
    _donors = gr.make_tensor_input("donors")
    weight_matrix = gr.make_tensor_input("weight_matrix")
    op = gr.get_op_builder(scope)
    _to_copy = op.Cast(donors_mask, to=itype, outputs=["_to_copy"])
    output_0 = op.Mul(_to_copy, weight_matrix, outputs=["output_0"])
    gr.make_tensor_output(output_0)
    g.make_local_function(
        container=gr,
        optimize=False,
        name="calc_impute_make_new_neights",
        domain="local_domain",
    )
    return gr


def make_calc_impute(g: ModelComponentContainer, scope: Scope, itype: int):
    gr = ModelComponentContainer(
        {"": g.main_opset, "local_domain": 1}, as_function=True
    )
    dist_pot_donors = gr.make_tensor_input("dist_pot_donors")
    n_neighbors = gr.make_tensor_input("n_neighbors")
    fit_x_col = gr.make_tensor_input("fit_x_col")
    mask_fit_x_col = gr.make_tensor_input("mask_fit_x_col")
    op = gr.get_op_builder(scope)
    c_lifted_tensor_0 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value"),
        outputs=["c_lifted_tensor_0"],
    )
    c_lifted_tensor_1 = op.Constant(
        value=from_array(
            np.array([1.0], dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["c_lifted_tensor_1"],
    )
    init7_s1__1 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value"),
        outputs=["init7_s1__1"],
    )
    init1_s_ = op.Constant(
        value=from_array(
            np.array(0.0, dtype=tensor_dtype_to_np_dtype(itype)), name="value"
        ),
        outputs=["init1_s_"],
    )
    (
        calc_impute_in__donors_idx__0,
        calc_impute_in__donors_idx__1,
    ) = op.calc_impute_donors(
        dist_pot_donors,
        n_neighbors,
        domain="local_domain",
        outputs=[
            "calc_impute_in__donors_idx__0",
            "calc_impute_in__donors_idx__1",
        ],
    )
    calc_impute_in__weights = op.calc_impute_weights(
        calc_impute_in__donors_idx__1,
        domain="local_domain",
        outputs=["calc_impute_in__weights"],
    )
    _reshape_fit_x_col0 = op.Reshape(
        fit_x_col, init7_s1__1, outputs=["_reshape_fit_x_col0"]
    )
    take = op.Gather(
        _reshape_fit_x_col0,
        calc_impute_in__donors_idx__0,
        outputs=["take"],
    )
    _reshape_mask_fit_x_col0 = op.Reshape(
        mask_fit_x_col, init7_s1__1, outputs=["_reshape_mask_fit_x_col0"]
    )
    take_1 = op.Gather(
        _reshape_mask_fit_x_col0,
        calc_impute_in__donors_idx__0,
        outputs=["take_1"],
    )
    _to_copy = op.Cast(take_1, to=TensorProto.INT64, outputs=["_to_copy"])
    sub_12 = op.Sub(c_lifted_tensor_0, _to_copy, outputs=["sub_12"])
    calc_impute_in__make_new_neights = op.calc_impute_make_new_neights(
        sub_12,
        take,
        calc_impute_in__weights,
        domain="local_domain",
        outputs=["calc_impute_in__make_new_neights"],
    )
    sum_1 = op.ReduceSumAnyOpset(
        calc_impute_in__make_new_neights,
        c_lifted_tensor_0,
        keepdims=1,
        outputs=["sum_1"],
    )
    _reshape_init1_s_0 = op.Reshape(
        init1_s_, c_lifted_tensor_0, outputs=["_reshape_init1_s_0"]
    )
    eq_17 = op.Equal(sum_1, _reshape_init1_s_0, outputs=["eq_17"])
    where = op.Where(eq_17, c_lifted_tensor_1, sum_1, outputs=["where"])
    mul_17 = op.Mul(
        take,
        calc_impute_in__make_new_neights,
        outputs=["mul_17"],
    )
    sum_2 = op.ReduceSumAnyOpset(
        mul_17, c_lifted_tensor_0, keepdims=1, outputs=["sum_2"]
    )
    div = op.Div(sum_2, where, outputs=["div"])
    output_0 = op.SqueezeAnyOpset(div, c_lifted_tensor_0, outputs=["output_0"])
    gr.make_tensor_output(output_0)
    g.make_local_function(
        container=gr, optimize=False, name="calc_impute", domain="local_domain"
    )
    return gr


def make_knn_imputer_column_all_nan(
    g: ModelComponentContainer, scope: Scope, itype: int
):
    """
    Sent:

    ::

        i_col,
        x,
        dist_subset,
        mask_fit_x,
        _fit_x,
        receivers_idx,
        all_nan_receivers_idx,
        all_nan_dist_mask,
        dist_chunk,
        dist_idx_map,
        potential_donors_idx,
    """
    gr = ModelComponentContainer(
        {"": g.main_opset, "local_domain": 1}, as_function=True
    )
    i_col = gr.make_tensor_input("i_col")
    x = gr.make_tensor_input("x")
    _dist_subset = gr.make_tensor_input("dist_subset")
    mask_fit_x = gr.make_tensor_input("mask_fit_x")
    _fit_x = gr.make_tensor_input("_fit_x")
    receivers_idx = gr.make_tensor_input("receivers_idx")
    all_nan_receivers_idx = gr.make_tensor_input("all_nan_receivers_idx")
    all_nan_dist_mask = gr.make_tensor_input("all_nan_dist_mask")
    dist_chunk = gr.make_tensor_input("dist_chunk")
    dist_idx_map = gr.make_tensor_input("dist_idx_map")
    potential_donors_idx = gr.make_tensor_input("potential_donors_idx")
    op = gr.get_op_builder(scope)
    c_lifted_tensor_0 = op.Constant(
        value=from_array(np.array([1.0], dtype=np.float32), name="value"),
        outputs=["c_lifted_tensor_0"],
    )
    # init7_s_2 = op.Constant(value=from_array(
    # np.array(2, dtype=np.int64), name='value'), outputs=['init7_s_2'])
    init7_s_2 = i_col
    init1_s_ = op.Constant(
        value=from_array(np.array(1.0, dtype=np.float32), name="value"),
        outputs=["init1_s_"],
    )
    init7_s1_1 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value"),
        outputs=["init7_s1_1"],
    )
    init1_s_2 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value"),
        outputs=["init1_s_2"],
    )
    init7_s1_0 = op.Constant(
        value=from_array(np.array([0], dtype=np.int64), name="value"),
        outputs=["init7_s1_0"],
    )
    init7_s1__1 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value"),
        outputs=["init7_s1__1"],
    )
    init7_s_1 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value"),
        outputs=["init7_s_1"],
    )
    # init7_s1_2 = op.Constant(value=from_array(np.array([2], dtype=np.int64),
    # name='value'), outputs=['init7_s1_2'])
    init7_s1_2 = op.UnsqueezeAnyOpset(i_col, init7_s1_0, outputs=["init7_s1_2"])
    select = op.Gather(mask_fit_x, init7_s_2, axis=1, outputs=["select"])
    bitwise_not = op.Not(select, outputs=["bitwise_not"])
    _to_copy = op.Cast(bitwise_not, to=1, outputs=["_to_copy"])
    sum_1 = op.ReduceSumAnyOpset(_to_copy, keepdims=0, outputs=["sum_1"])
    _reshape_init1_s_0 = op.Reshape(
        init1_s_, init7_s1_1, outputs=["_reshape_init1_s_0"]
    )
    eq_6 = op.Equal(_to_copy, _reshape_init1_s_0, outputs=["eq_6"])
    select_1 = op.Gather(_fit_x, init7_s_2, axis=1, outputs=["select_1"])
    index = op.Compress(select_1, eq_6, axis=0, outputs=["index"])
    sum_2 = op.ReduceSumAnyOpset(index, keepdims=0, outputs=["sum_2"])
    gt = op.Greater(sum_1, init1_s_2, outputs=["gt"])
    where = op.Where(gt, sum_1, c_lifted_tensor_0, outputs=["where"])
    _reshape__to_copy_20 = op.Reshape(
        sum_2, init7_s1_1, outputs=["_reshape__to_copy_20"]
    )
    div = op.Div(_reshape__to_copy_20, where, outputs=["div"])
    select_2 = op.Gather(x, init7_s_2, axis=1, outputs=["select_2"])
    view = op.SqueezeAnyOpset(div, init7_s1_0, outputs=["view"])
    _onx_unsqueeze_all_nan_receivers_idx0 = op.UnsqueezeAnyOpset(
        all_nan_receivers_idx,
        init7_s1__1,
        outputs=["_onx_unsqueeze_all_nan_receivers_idx0"],
    )
    _shape_all_nan_receivers_idx0 = op.Shape(
        all_nan_receivers_idx, outputs=["_shape_all_nan_receivers_idx0"]
    )
    _onx_expand_view0 = op.Expand(
        view, _shape_all_nan_receivers_idx0, outputs=["_onx_expand_view0"]
    )
    index_put = op.ScatterND(
        select_2,
        _onx_unsqueeze_all_nan_receivers_idx0,
        _onx_expand_view0,
        outputs=["index_put"],
    )
    _onx_unsqueeze_index_put0 = op.UnsqueezeAnyOpset(
        index_put, init7_s_1, outputs=["_onx_unsqueeze_index_put0"]
    )
    _shape_unsqueeze_index_put00 = op.Shape(
        _onx_unsqueeze_index_put0, outputs=["_shape_unsqueeze_index_put00"]
    )
    _onx_expand_init7_s1_20 = op.Expand(
        init7_s1_2, _shape_unsqueeze_index_put00, outputs=["_onx_expand_init7_s1_20"]
    )
    output_0 = op.ScatterElements(
        x,
        _onx_expand_init7_s1_20,
        _onx_unsqueeze_index_put0,
        axis=1,
        reduction="none",
        outputs=["output_0"],
    )
    bitwise_not_1 = op.Not(all_nan_dist_mask, outputs=["bitwise_not_1"])
    output_2 = op.Compress(receivers_idx, bitwise_not_1, axis=0, outputs=["output_2"])
    index_2 = op.Gather(dist_idx_map, output_2, axis=0, outputs=["index_2"])
    index_3 = op.Gather(dist_chunk, index_2, axis=0, outputs=["index_3"])
    output_1 = op.Gather(index_3, potential_donors_idx, axis=1, outputs=["output_1"])
    gr.make_tensor_output(output_0)
    gr.make_tensor_output(output_1)
    gr.make_tensor_output(output_2)

    g.make_local_function(
        container=gr,
        optimize=False,
        name="knn_imputer_column_all_nan",
        domain="local_domain",
    )
    return gr


def make_knn_imputer_column_nan_found(
    g: ModelComponentContainer, scope: Scope, itype: int
):
    gr = ModelComponentContainer(
        {"": g.main_opset, "local_domain": 1}, as_function=True
    )
    i_col = gr.make_tensor_input("i_col")
    x = gr.make_tensor_input("x")
    col_mask = gr.make_tensor_input("col_mask")
    dist_idx_map = gr.make_tensor_input("dist_idx_map")
    mask_fit_x = gr.make_tensor_input("mask_fit_x")
    non_missing_fix_x = gr.make_tensor_input("non_missing_fix_x")
    dist_chunk = gr.make_tensor_input("dist_chunk")
    _fit_x = gr.make_tensor_input("_fit_x")
    row_missing_idx = gr.make_tensor_input("row_missing_idx")
    n_neighours = gr.make_tensor_input("n_neighbors")

    op = gr.get_op_builder(scope)

    init7_s_2 = i_col
    init7_s1__1 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value"),
        outputs=["init7_s1__1"],
    )
    init7_s_0 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value"),
        outputs=["init7_s_0"],
    )
    init7_s_1 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value"),
        outputs=["init7_s_1"],
    )
    c_lifted_tensor_2 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value"),
        outputs=["c_lifted_tensor_2"],
    )

    init7_s1_2 = op.UnsqueezeAnyOpset(init7_s_2, np.array([0], dtype=np.int64))

    select_1 = op.Gather(non_missing_fix_x, init7_s_2, axis=1, outputs=["select_1"])
    potential_donors_idx = op.NonZero(select_1, outputs=["potential_donors_idx"])

    nonzero_numpy__0 = op.Reshape(
        potential_donors_idx, init7_s1__1, outputs=["nonzero_numpy__0"]
    )
    _shape_getitem_20 = op.Shape(
        nonzero_numpy__0, end=1, start=0, outputs=["_shape_getitem_20"]
    )
    sym_size_int_23 = op.SqueezeAnyOpset(_shape_getitem_20, outputs=["sym_size_int_23"])

    flat_index = op.NonZero(col_mask, outputs=["flat_index"])
    nonzero_numpy_1__0 = op.Reshape(
        flat_index, init7_s1__1, outputs=["nonzero_numpy_1__0"]
    )
    receivers_idx = op.Gather(
        row_missing_idx, nonzero_numpy_1__0, axis=0, outputs=["receivers_idx"]
    )
    index_2 = op.Gather(dist_idx_map, receivers_idx, axis=0, outputs=["index_2"])
    index_3 = op.Gather(dist_chunk, index_2, axis=0, outputs=["index_3"])
    dist_subset = op.Gather(index_3, nonzero_numpy__0, axis=1, outputs=["dist_subset"])
    isnan = op.IsNaN(dist_subset, outputs=["isnan"])
    _onx_cast_isnan0 = op.Cast(isnan, to=6, outputs=["_onx_cast_isnan0"])
    _onx_reducemin_cast_isnan00 = op.ReduceMinAnyOpset(
        _onx_cast_isnan0,
        c_lifted_tensor_2,
        keepdims=0,
        outputs=["_onx_reducemin_cast_isnan00"],
    )
    all_nan_dist_mask = op.Cast(
        _onx_reducemin_cast_isnan00, to=TensorProto.BOOL, outputs=["all_nan_dist_mask"]
    )
    all_nan_receivers_idx = op.Compress(
        receivers_idx, all_nan_dist_mask, axis=0, outputs=["all_nan_receivers_idx"]
    )

    size_all_nan_receivers_idx = op.Size(all_nan_receivers_idx)
    zero_i = op.Constant(value=from_array(np.array(0, dtype=np.int64), name="zero"))
    size_index_5_greater = op.Greater(size_all_nan_receivers_idx, zero_i)
    select_scatter, dist_subset, receivers_idx = op.If(
        size_index_5_greater,
        then_branch=make_graph(
            [
                make_node(
                    "knn_imputer_column_all_nan",
                    [
                        i_col,
                        x,
                        dist_subset,
                        mask_fit_x,
                        _fit_x,
                        receivers_idx,
                        all_nan_receivers_idx,
                        all_nan_dist_mask,
                        dist_chunk,
                        dist_idx_map,
                        potential_donors_idx,
                    ],
                    ["A", "B", "C"],
                    domain="local_domain",
                )
            ],
            "then_branch",
            [],
            [
                make_tensor_value_info("A", itype, None),
                make_tensor_value_info("B", itype, None),
                make_tensor_value_info("C", TensorProto.INT64, None),
            ],
        ),
        else_branch=make_graph(
            [
                make_node("Identity", [x], ["A"]),
                make_node("Identity", [dist_subset], ["B"]),
                make_node("Identity", [receivers_idx], ["C"]),
            ],
            "identity",
            [],
            [
                make_tensor_value_info("A", itype, None),
                make_tensor_value_info("B", itype, None),
                make_tensor_value_info("C", TensorProto.INT64, None),
            ],
        ),
        outputs=["select_scatter", "dist_subset_updated", "receivers_idx_updated"],
    )

    lt = op.Less(n_neighours, sym_size_int_23, outputs=["lt"])
    where_1 = op.Where(lt, n_neighours, sym_size_int_23, outputs=["where_1"])
    le = op.LessOrEqual(where_1, init7_s_0, outputs=["le"])
    # c_lifted_tensor_2 -> init7_s_1 to have a zero time, onnxruntime crashes wher shapes are
    # (), (1,), ()
    where_2 = op.Where(le, init7_s_1, where_1, outputs=["where_2"])
    select_6 = op.Gather(_fit_x, init7_s_2, axis=1, outputs=["select_6"])
    index_11 = op.Gather(select_6, nonzero_numpy__0, axis=0, outputs=["index_11"])
    select_7 = op.Gather(mask_fit_x, init7_s_2, axis=1, outputs=["select_7"])
    index_12 = op.Gather(select_7, nonzero_numpy__0, axis=0, outputs=["index_12"])
    calc_impute_output = op.calc_impute(
        dist_subset,
        where_2,
        index_11,
        index_12,
        domain="local_domain",
        outputs=["calc_impute_output"],
    )
    select_9 = op.Gather(select_scatter, init7_s_2, axis=1, outputs=["select_9"])
    _onx_unsqueeze_index_70 = op.UnsqueezeAnyOpset(
        receivers_idx, init7_s1__1, outputs=["_onx_unsqueeze_index_70"]
    )
    index_put_1 = op.ScatterND(
        select_9,
        _onx_unsqueeze_index_70,
        calc_impute_output,
        outputs=["index_put_1"],
    )
    _onx_unsqueeze_index_put_10 = op.UnsqueezeAnyOpset(
        index_put_1, init7_s_1, outputs=["_onx_unsqueeze_index_put_10"]
    )
    _shape_unsqueeze_index_put_100 = op.Shape(
        _onx_unsqueeze_index_put_10, outputs=["_shape_unsqueeze_index_put_100"]
    )
    _onx_expand_init7_s1_202 = op.Expand(
        init7_s1_2, _shape_unsqueeze_index_put_100, outputs=["_onx_expand_init7_s1_202"]
    )
    output_0 = op.ScatterElements(
        select_scatter,
        _onx_expand_init7_s1_202,
        _onx_unsqueeze_index_put_10,
        axis=1,
        reduction="none",
        outputs=["output_0"],
    )
    gr.make_tensor_output(output_0)
    g.make_local_function(
        container=gr,
        optimize=False,
        name="knn_imputer_column_nan_found",
        domain="local_domain",
    )
    return gr


def make_knn_imputer_column(g: ModelComponentContainer, scope: Scope, itype: int):
    gr = ModelComponentContainer(
        {"": g.main_opset, "local_domain": 1}, as_function=True
    )
    i_col = gr.make_tensor_input("i_col")
    x = gr.make_tensor_input("x")
    dist_chunk = gr.make_tensor_input("dist_chunk")
    non_missing_fix_x = gr.make_tensor_input("non_missing_fix_x")
    mask_fit_x = gr.make_tensor_input("mask_fit_x")
    dist_idx_map = gr.make_tensor_input("dist_idx_map")
    mask = gr.make_tensor_input("mask")
    row_missing_idx = gr.make_tensor_input("row_missing_idx")
    _fit_x = gr.make_tensor_input("_fit_x")
    n_neighbors = gr.make_tensor_input("n_neighbors")

    op = gr.get_op_builder(scope)
    zero32 = op.Constant(
        value=from_array(np.array(0, dtype=np.int32), name="value"),
        outputs=["zero"],
    )
    init7_s_2 = i_col
    # init7_s_2 = op.Constant(value=from_array(np.array(2, dtype=np.int64), name='value'),
    # outputs=['init7_s_2'])
    init7_s1__1 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value"),
        outputs=["init7_s1__1"],
    )
    # init7_s1_2 = op.Constant(value=from_array(np.array([2], dtype=np.int64), name='value'),
    # outputs=['init7_s1_2'])
    select = op.Gather(mask, init7_s_2, axis=1, outputs=["select"])
    index = op.Gather(select, row_missing_idx, axis=0, outputs=["index"])
    view = op.Reshape(index, init7_s1__1, outputs=["view"])
    col_mask = op.Cast(view, to=TensorProto.INT32)

    # If max(col_mask) == 0, no need to continue
    # That avoids dealing with empty sizes.
    col_mask_max = op.ReduceMax(col_mask, outputs=["col_mask_max"])
    col_mask_max_null = op.Equal(col_mask_max, zero32, outputs=["col_mask_max_null"])

    output_0 = op.If(
        col_mask_max_null,
        then_branch=make_graph(
            [make_node("Identity", [x], ["X"])],
            "identity",
            [],
            [make_tensor_value_info("X", itype, None)],
        ),
        else_branch=make_graph(
            [
                make_node(
                    "knn_imputer_column_nan_found",
                    [
                        i_col,
                        x,
                        col_mask,
                        dist_idx_map,
                        mask_fit_x,
                        non_missing_fix_x,
                        dist_chunk,
                        _fit_x,
                        row_missing_idx,
                        n_neighbors,
                    ],
                    ["X"],
                    domain="local_domain",
                )
            ],
            "then_branch",
            [],
            [make_tensor_value_info("X", itype, None)],
        ),
        outputs=["output_0"],
    )
    gr.make_tensor_output(output_0)
    g.make_local_function(
        container=gr, optimize=False, name="knn_imputer_column", domain="local_domain"
    )
    return gr


def _knn_imputer_builder(
    op: ModelComponentContainer,
    _mask_fit_x: "BOOL[s0, 2]",  # noqa: F821
    _valid_mask: "BOOL[2]",  # noqa: F821
    _fit_x: "DOUBLE[s1, 2]",  # noqa: F821
    n_neighbors: "INT",  # noqa: F821
    x: "FLOAT[s2, 2]",  # noqa: F821
    itype: int,
):
    n_cols = _fit_x.shape[1]
    init7_s1_1 = np.array([1], dtype=np.int64)
    init7_s1__1 = np.array([-1], dtype=np.int64)
    isnan = op.IsNaN(x, outputs=["isnan"])
    _onx_compress_isnan0 = op.Compress(
        isnan, _valid_mask, axis=1, outputs=["_onx_compress_isnan0"]
    )
    _onx_cast_index0 = op.Cast(_onx_compress_isnan0, to=6, outputs=["_onx_cast_index0"])
    _onx_reducemax_cast_index00 = op.ReduceMaxAnyOpset(
        _onx_cast_index0,
        init7_s1_1,
        keepdims=0,
        outputs=["_onx_reducemax_cast_index00"],
    )
    any_1 = op.Cast(_onx_reducemax_cast_index00, to=TensorProto.BOOL, outputs=["any_1"])
    view = op.Reshape(any_1, init7_s1__1, outputs=["view"])
    _onx_nonzero_view0 = op.NonZero(view, outputs=["_onx_nonzero_view0"])
    nonzero_numpy__0 = op.Reshape(
        _onx_nonzero_view0, init7_s1__1, outputs=["nonzero_numpy__0"]
    )
    logical_not = op.Not(_mask_fit_x, outputs=["logical_not"])
    make_dict_idx_map = op.dict_idx_map(
        x, nonzero_numpy__0, domain="local_domain", outputs=["make_dict_idx_map"]
    )
    index_1 = op.Gather(x, nonzero_numpy__0, axis=0, outputs=["index_1"])
    c_torch_knnimputer_dist = op.dist_nan_euclidean(
        index_1, _fit_x, domain="local_domain", outputs=["c_torch_knnimputer_dist"]
    )
    for i_cols in range(n_cols):
        x = op.knn_imputer_column(
            np.array(i_cols, dtype=np.int64),
            x,
            c_torch_knnimputer_dist,
            logical_not,
            _mask_fit_x,
            make_dict_idx_map,
            isnan,
            nonzero_numpy__0,
            _fit_x,
            n_neighbors,
            domain="local_domain",
        )
    return op.Compress(x, _valid_mask, axis=1, outputs=["output_0"])


def convert_knn_imputer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNNImputer* into *ONNX*.
    """
    assert (
        container.target_opset >= 18
    ), f"This converter no longer works for opset {container.target_opset} < 18."
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT
    knn_op = operator.raw_operator
    if knn_op.metric != "nan_euclidean":
        raise RuntimeError("Unable to convert KNNImputer when metric is callable.")
    if knn_op.weights not in ("uniform", "distance"):
        raise RuntimeError(
            f"Unable to convert KNNImputer when weights "
            f"is callable, knn_op.weights={knn_op.weights}"
        )
    if knn_op.weights == "distance":
        raise NotImplementedError(
            "KNNImputer with distance as metric is not supported, "
            "you may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )
    # options = container.get_options(knn_op, dict(optim=None))
    # options are not used anymore
    training_data = knn_op._fit_X.astype(dtype)
    result = _knn_imputer_builder(
        container.get_op_builder(scope),
        knn_op._mask_fit_X,
        knn_op._valid_mask,
        training_data,
        np.array(knn_op.n_neighbors, dtype=np.int64),
        operator.inputs[0].full_name,
        itype=proto_type,
    )
    make_dict_idx_map(container, Scope("LF1"), itype=proto_type)
    make_dist_nan_euclidean(container, Scope("LF2"), itype=proto_type)
    make_calc_impute_donors(container, Scope("LF3"), itype=proto_type)
    make_calc_impute_weights(container, Scope("LF4"), itype=proto_type)
    make_calc_impute_make_new_neights(container, Scope("LF5"), itype=proto_type)
    make_calc_impute(container, Scope("LF6"), itype=proto_type)
    make_knn_imputer_column_all_nan(container, Scope("LF7"), itype=proto_type)
    make_knn_imputer_column_nan_found(container, Scope("LF8"), itype=proto_type)
    make_knn_imputer_column(container, Scope("LF9"), itype=proto_type)
    container.add_node("Identity", [result], [operator.outputs[0].full_name])


def convert_nca(scope: Scope, operator: Operator, container: ModelComponentContainer):
    """
    Converts *NeighborhoodComponentsAnalysis* into *ONNX*.
    """
    X = operator.inputs[0]
    nca_op = operator.raw_operator
    op_version = container.target_opset
    out = operator.outputs
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    components = nca_op.components_.T.astype(dtype)

    if isinstance(X.type, Int64TensorType):
        X = OnnxCast(X, to=onnx_proto.TensorProto.FLOAT, op_version=op_version)
    elif isinstance(X.type, DoubleTensorType):
        components = OnnxCast(
            components, to=onnx_proto.TensorProto.DOUBLE, op_version=op_version
        )
    else:
        components = components.astype(dtype)
    res = OnnxMatMul(X, components, output_names=out[:1], op_version=op_version)
    res.add_to(scope, container)


register_converter(
    "SklearnKNeighborsClassifier",
    convert_nearest_neighbors_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "raw_scores": [True, False],
        "output_class_labels": [False, True],
        "optim": [None, "cdist"],
    },
)
register_converter(
    "SklearnRadiusNeighborsClassifier",
    convert_nearest_neighbors_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "raw_scores": [True, False],
        "output_class_labels": [False, True],
        "optim": [None, "cdist"],
    },
)
register_converter(
    "SklearnKNeighborsRegressor",
    convert_nearest_neighbors_regressor,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnRadiusNeighborsRegressor",
    convert_nearest_neighbors_regressor,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnKNeighborsTransformer",
    convert_k_neighbours_transformer,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnNearestNeighbors",
    convert_nearest_neighbors_transform,
    options={"optim": [None, "cdist"]},
)
register_converter(
    "SklearnKNNImputer", convert_knn_imputer, options={"optim": [None, "cdist"]}
)
register_converter("SklearnNeighborhoodComponentsAnalysis", convert_nca)
