# SPDX-License-Identifier: Apache-2.0


import numpy as np
from onnx.helper import make_tensor
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
    tensor_value = py_make_float_array(-1, dtype=np.float32, as_tensor=True)
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


def _knn_imputer_builder(
    op: "GraphBuilder",  #  noqa: F821
    _mask_fit_x: "BOOL[:, C]",  #  noqa: F821
    _valid_mask: "BOOL[C]",  #  noqa: F821
    _fit_x: "DOUBLE[:,C]",  #  noqa: F821
    x: "FLOAT[:,C]",  #  noqa: F821
):
    init7_s1_1 = np.array([1], dtype=np.int64)
    init7_s1__1 = np.array([-1], dtype=np.int64)
    isnan = op.IsNaN(x)
    _onx_compress_isnan0 = op.Compress(isnan, _valid_mask, axis=1)
    _onx_cast_index0 = op.Cast(_onx_compress_isnan0, to=6)
    _onx_reducemax_cast_index00 = op.ReduceMaxAnyOpset(
        _onx_cast_index0, init7_s1_1, keepdims=0
    )
    any_1 = op.Cast(_onx_reducemax_cast_index00, to=9)
    view = op.Reshape(any_1, init7_s1__1)
    _onx_nonzero_view0 = op.NonZero(view)
    nonzero_numpy__0 = op.Reshape(_onx_nonzero_view0, init7_s1__1)
    logical_not = op.Not(_mask_fit_x)
    init7_s_0__1 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value")
    )
    init7_s_4__1 = op.Constant(
        value=from_array(np.array(4, dtype=np.int64), name="value")
    )
    init7_s_1__1 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value")
    )
    init7_s1__1__1 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value")
    )
    _shape_x0__1 = op.Shape(x, end=1, start=0)
    zeros__1 = op.ConstantOfShape(
        _shape_x0__1, value=from_array(np.array([0], dtype=np.int64), name="value")
    )
    arange__1_0 = op.Range(init7_s_0__1, init7_s_4__1, init7_s_1__1)
    _onx_unsqueeze_row_missing_idx0__1 = op.UnsqueezeAnyOpset(
        nonzero_numpy__0, init7_s1__1__1
    )
    c_torch_knnimputer__make_dict_idx_map = op.ScatterND(
        zeros__1, _onx_unsqueeze_row_missing_idx0__1, arange__1_0
    )
    index_1 = op.Gather(x, nonzero_numpy__0, axis=0)
    c_lifted_tensor_0__2 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value")
    )
    c_lifted_tensor_1__2 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value")
    )
    c_lifted_tensor_2__2 = op.Constant(
        value=from_array(np.array(np.nan, dtype=np.float32), name="value")
    )
    c_lifted_tensor_3__2 = op.Constant(
        value=from_array(np.array([1.0], dtype=np.float32), name="value")
    )
    init1_s___2 = op.Constant(
        value=from_array(np.array(-2.0, dtype=np.float32), name="value")
    )
    init7_s1_1__2 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value")
    )
    init1_s1___2 = op.Constant(
        value=from_array(np.array([0.0], dtype=np.float32), name="value")
    )
    init1_s_2__2 = op.Constant(
        value=from_array(np.array(1.0, dtype=np.float32), name="value")
    )
    init1_s_3__2 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value")
    )
    init1_s_4__2 = op.Constant(
        value=from_array(np.array(2.0, dtype=np.float32), name="value")
    )
    init7_s2_1__1__2 = op.Constant(
        value=from_array(np.array([1, -1], dtype=np.int64), name="value")
    )
    _to_copy__2_1 = op.Cast(_fit_x, to=1)
    isnan__2_2 = op.IsNaN(index_1)
    _to_copy_2__2 = op.Cast(isnan__2_2, to=1)
    isnan_1__2 = op.IsNaN(_to_copy__2_1)
    index_put__2 = op.Where(isnan__2_2, c_lifted_tensor_0__2, index_1)
    index_put_1__2 = op.Where(isnan_1__2, c_lifted_tensor_1__2, _to_copy__2_1)
    mul_10__2 = op.Mul(index_put__2, index_put__2)
    mul_11__2 = op.Mul(index_put_1__2, index_put_1__2)
    matmul_2__2 = op.Gemm(_to_copy_2__2, mul_11__2, transA=0, transB=1)
    _reshape_init1_s_0__2 = op.Reshape(init1_s___2, init7_s1_1__2)
    _onx_mul_index_put0__2 = op.Mul(index_put__2, _reshape_init1_s_0__2)
    matmul__2 = op.Gemm(_onx_mul_index_put0__2, index_put_1__2, transA=0, transB=1)
    sum_1__2 = op.ReduceSumAnyOpset(mul_10__2, init7_s1_1__2, keepdims=1)
    add_26__2 = op.Add(matmul__2, sum_1__2)
    sum_2__2 = op.ReduceSumAnyOpset(mul_11__2, init7_s1_1__2, keepdims=1)
    permute_2__2 = op.Reshape(sum_2__2, init7_s2_1__1__2)
    add_35__2 = op.Add(add_26__2, permute_2__2)
    _to_copy_1__2 = op.Cast(isnan_1__2, to=1)
    matmul_1__2 = op.Gemm(mul_10__2, _to_copy_1__2, transA=0, transB=1)
    sub_18__2 = op.Sub(add_35__2, matmul_1__2)
    sub_24__2 = op.Sub(sub_18__2, matmul_2__2)
    clip__2_3 = op.Clip(sub_24__2, init1_s1___2)
    _reshape_init1_s_20__2 = op.Reshape(init1_s_2__2, init7_s1_1__2)
    rsub__2 = op.Sub(_reshape_init1_s_20__2, _to_copy_2__2)
    bitwise_not__2_4 = op.Not(isnan_1__2)
    _to_copy_4__2 = op.Cast(bitwise_not__2_4, to=1)
    matmul_3__2 = op.Gemm(rsub__2, _to_copy_4__2, transA=0, transB=1)
    _reshape_init1_s_30__2 = op.Reshape(init1_s_3__2, init7_s1_1__2)
    eq_33__2 = op.Equal(matmul_3__2, _reshape_init1_s_30__2)
    index_put_2__2 = op.Where(eq_33__2, c_lifted_tensor_2__2, clip__2_3)
    maximum__2_5 = op.Max(c_lifted_tensor_3__2, matmul_3__2)
    div__2 = op.Div(index_put_2__2, maximum__2_5)
    _reshape_init1_s_40__2 = op.Reshape(init1_s_4__2, init7_s1_1__2)
    _onx_mul_div0__2 = op.Mul(div__2, _reshape_init1_s_40__2)
    c_torch_knnimputer_dist = op.Sqrt(_onx_mul_div0__2)
    c_lifted_tensor_0__3 = op.Constant(
        value=from_array(np.array([1.0], dtype=np.float32), name="value")
    )
    c_lifted_tensor_1__3 = op.Constant(
        value=from_array(np.array(3, dtype=np.int64), name="value")
    )
    c_lifted_tensor_2__3 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value")
    )
    init7_s_0__3 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value")
    )
    init7_s1__1__3 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value")
    )
    init11_s___3 = op.Constant(
        value=from_array(np.array(1.0, dtype=np.float64), name="value")
    )
    init1_s___3 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value")
    )
    init7_s1_0__3 = op.Constant(
        value=from_array(np.array([0], dtype=np.int64), name="value")
    )
    init7_s_1__3 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value")
    )
    select__3 = op.Gather(isnan, init7_s_0__3, axis=1)
    index__3 = op.Gather(select__3, nonzero_numpy__0, axis=0)
    select_1__3 = op.Gather(logical_not, init7_s_0__3, axis=1)
    _onx_nonzero_select_10__3 = op.NonZero(select_1__3)
    nonzero_numpy__0__3 = op.Reshape(_onx_nonzero_select_10__3, init7_s1__1__3)
    _shape_getitem_20__3 = op.Shape(nonzero_numpy__0__3, end=1, start=0)
    sym_size_int_20__3 = op.SqueezeAnyOpset(_shape_getitem_20__3)
    view__3_6 = op.Reshape(index__3, init7_s1__1__3)
    _onx_nonzero_view0__3 = op.NonZero(view__3_6)
    nonzero_numpy_1__0__3 = op.Reshape(_onx_nonzero_view0__3, init7_s1__1__3)
    index_1__3 = op.Gather(nonzero_numpy__0, nonzero_numpy_1__0__3, axis=0)
    index_2__3 = op.Gather(c_torch_knnimputer__make_dict_idx_map, index_1__3, axis=0)
    index_3__3 = op.Gather(c_torch_knnimputer_dist, index_2__3, axis=0)
    _onx_gather_index_30__3 = op.Gather(index_3__3, nonzero_numpy__0__3, axis=1)
    isnan__3_7 = op.IsNaN(_onx_gather_index_30__3)
    _onx_cast_isnan0__3 = op.Cast(isnan__3_7, to=6)
    _onx_reducemin_cast_isnan00__3 = op.ReduceMinAnyOpset(
        _onx_cast_isnan0__3, c_lifted_tensor_2__3, keepdims=0
    )
    all_1__3 = op.Cast(_onx_reducemin_cast_isnan00__3, to=9)
    index_5__3 = op.Compress(index_1__3, all_1__3, axis=0)
    select_2__3 = op.Gather(_mask_fit_x, init7_s_0__3, axis=1)
    bitwise_not__3_8 = op.Not(select_2__3)
    _to_copy__3_9 = op.Cast(bitwise_not__3_8, to=11)
    _to_copy_1__3 = op.Cast(_to_copy__3_9, to=1)
    sum_1__3 = op.ReduceSumAnyOpset(_to_copy_1__3, keepdims=0)
    _reshape_init11_s_0__3 = op.Reshape(init11_s___3, c_lifted_tensor_2__3)
    eq_23__3 = op.Equal(_to_copy__3_9, _reshape_init11_s_0__3)
    select_3__3 = op.Gather(_fit_x, init7_s_0__3, axis=1)
    index_6__3 = op.Compress(select_3__3, eq_23__3, axis=0)
    sum_2__3 = op.ReduceSumAnyOpset(index_6__3, keepdims=0)
    _to_copy_2__3 = op.Cast(sum_2__3, to=1)
    gt__3 = op.Greater(sum_1__3, init1_s___3)
    where__3 = op.Where(gt__3, sum_1__3, c_lifted_tensor_0__3)
    _reshape__to_copy_20__3 = op.Reshape(_to_copy_2__3, c_lifted_tensor_2__3)
    div__3 = op.Div(_reshape__to_copy_20__3, where__3)
    select_4__3 = op.Gather(x, init7_s_0__3, axis=1)
    view_1__3 = op.SqueezeAnyOpset(div__3, init7_s1_0__3)
    _onx_unsqueeze_index_50__3 = op.UnsqueezeAnyOpset(index_5__3, init7_s1__1__3)
    _shape_index_502__3 = op.Shape(index_5__3)
    _onx_expand_view_10__3 = op.Expand(view_1__3, _shape_index_502__3)
    index_put__3 = op.ScatterND(
        select_4__3, _onx_unsqueeze_index_50__3, _onx_expand_view_10__3
    )
    _onx_unsqueeze_index_put0__3 = op.UnsqueezeAnyOpset(index_put__3, init7_s_1__3)
    _shape_unsqueeze_index_put00__3 = op.Shape(_onx_unsqueeze_index_put0__3)
    _onx_expand_init7_s1_00__3 = op.Expand(
        init7_s1_0__3, _shape_unsqueeze_index_put00__3
    )
    select_scatter__3_10 = op.ScatterElements(
        x,
        _onx_expand_init7_s1_00__3,
        _onx_unsqueeze_index_put0__3,
        axis=1,
        reduction="none",
    )
    bitwise_not_1__3 = op.Not(all_1__3)
    index_7__3 = op.Compress(index_1__3, bitwise_not_1__3, axis=0)
    index_8__3 = op.Gather(c_torch_knnimputer__make_dict_idx_map, index_7__3, axis=0)
    index_9__3 = op.Gather(c_torch_knnimputer_dist, index_8__3, axis=0)
    _onx_gather_index_90__3 = op.Gather(index_9__3, nonzero_numpy__0__3, axis=1)
    lt__3 = op.Less(c_lifted_tensor_1__3, sym_size_int_20__3)
    where_1__3 = op.Where(lt__3, c_lifted_tensor_1__3, sym_size_int_20__3)
    le_3__3 = op.LessOrEqual(where_1__3, init7_s_0__3)
    where_2__3 = op.Where(le_3__3, c_lifted_tensor_2__3, where_1__3)
    select_6__3 = op.Gather(_fit_x, init7_s_0__3, axis=1)
    index_11__3 = op.Gather(select_6__3, nonzero_numpy__0__3, axis=0)
    select_7__3 = op.Gather(_mask_fit_x, init7_s_0__3, axis=1)
    index_12__3 = op.Gather(select_7__3, nonzero_numpy__0__3, axis=0)
    c_lifted_tensor_0__4 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value")
    )
    c_lifted_tensor_1__4 = op.Constant(
        value=from_array(np.array([1.0], dtype=np.float64), name="value")
    )
    init7_s1__1__4 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value")
    )
    init11_s___4 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float64), name="value")
    )
    init7_s_0__5 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value")
    )
    init7_s1_1__5 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value")
    )
    init7_s_1__5 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value")
    )
    _shape_dist_pot_donors0__5 = op.Shape(_onx_gather_index_90__3, end=1, start=0)
    sym_size_int_4__5 = op.SqueezeAnyOpset(_shape_dist_pot_donors0__5)
    (
        unused_topk_values__5,
        c_torch_knnimputer_columns_0___calc_impute__donors_idx__0__4,
    ) = op.TopK(_onx_gather_index_90__3, where_2__3, largest=0, sorted=1)
    arange__5_13 = op.Range(init7_s_0__5, sym_size_int_4__5, init7_s_1__5)
    unsqueeze__5_14 = op.UnsqueezeAnyOpset(arange__5_13, init7_s1_1__5)
    _onx_gathernd_dist_pot_donors0__5 = op.GatherND(
        _onx_gather_index_90__3, unsqueeze__5_14, batch_dims=0
    )
    c_torch_knnimputer_columns_0___calc_impute__donors_idx__1__4 = op.GatherElements(
        _onx_gathernd_dist_pot_donors0__5,
        c_torch_knnimputer_columns_0___calc_impute__donors_idx__0__4,
        axis=1,
    )
    c_lifted_tensor_0__6 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value")
    )
    _shape_donors_dist0__6 = op.Shape(
        c_torch_knnimputer_columns_0___calc_impute__donors_idx__1__4
    )
    ones_like__6_15 = op.ConstantOfShape(
        _shape_donors_dist0__6,
        value=from_array(np.array([1.0], dtype=np.float32), name="value"),
    )
    isnan__6_16 = op.IsNaN(c_torch_knnimputer_columns_0___calc_impute__donors_idx__1__4)
    c_torch_knnimputer_columns_0___calc_impute__weights__4 = op.Where(
        isnan__6_16, c_lifted_tensor_0__6, ones_like__6_15
    )
    _reshape_fit_x_col0__4 = op.Reshape(index_11__3, init7_s1__1__4)
    take__4_11 = op.Gather(
        _reshape_fit_x_col0__4,
        c_torch_knnimputer_columns_0___calc_impute__donors_idx__0__4,
    )
    _reshape_mask_fit_x_col0__4 = op.Reshape(index_12__3, init7_s1__1__4)
    take_1__4 = op.Gather(
        _reshape_mask_fit_x_col0__4,
        c_torch_knnimputer_columns_0___calc_impute__donors_idx__0__4,
    )
    _to_copy__4_12 = op.Cast(take_1__4, to=7)
    sub_12__4 = op.Sub(c_lifted_tensor_0__4, _to_copy__4_12)
    _to_copy__7_17 = op.Cast(sub_12__4, to=11)
    _to_copy_1__7 = op.Cast(
        c_torch_knnimputer_columns_0___calc_impute__weights__4, to=11
    )
    c_torch_knnimputer_columns_0___calc_impute__make_new_neights__4 = op.Mul(
        _to_copy__7_17, _to_copy_1__7
    )
    sum_1__4 = op.ReduceSumAnyOpset(
        c_torch_knnimputer_columns_0___calc_impute__make_new_neights__4,
        c_lifted_tensor_0__4,
        keepdims=1,
    )
    _reshape_init11_s_0__4 = op.Reshape(init11_s___4, c_lifted_tensor_0__4)
    eq_17__4 = op.Equal(sum_1__4, _reshape_init11_s_0__4)
    where__4 = op.Where(eq_17__4, c_lifted_tensor_1__4, sum_1__4)
    mul_17__4 = op.Mul(
        take__4_11, c_torch_knnimputer_columns_0___calc_impute__make_new_neights__4
    )
    sum_2__4 = op.ReduceSumAnyOpset(mul_17__4, c_lifted_tensor_0__4, keepdims=1)
    div__4 = op.Div(sum_2__4, where__4)
    _onx_squeeze_div0__4 = op.SqueezeAnyOpset(div__4, c_lifted_tensor_0__4)
    c_torch_knnimputer_columns_0___calc_impute__3 = op.Cast(_onx_squeeze_div0__4, to=1)
    select_9__3 = op.Gather(select_scatter__3_10, init7_s_0__3, axis=1)
    _onx_unsqueeze_index_70__3 = op.UnsqueezeAnyOpset(index_7__3, init7_s1__1__3)
    index_put_1__3 = op.ScatterND(
        select_9__3,
        _onx_unsqueeze_index_70__3,
        c_torch_knnimputer_columns_0___calc_impute__3,
    )
    _onx_unsqueeze_index_put_10__3 = op.UnsqueezeAnyOpset(index_put_1__3, init7_s_1__3)
    _shape_unsqueeze_index_put_100__3 = op.Shape(_onx_unsqueeze_index_put_10__3)
    _onx_expand_init7_s1_002__3 = op.Expand(
        init7_s1_0__3, _shape_unsqueeze_index_put_100__3
    )
    c_torch_knnimputer_columns_0_ = op.ScatterElements(
        select_scatter__3_10,
        _onx_expand_init7_s1_002__3,
        _onx_unsqueeze_index_put_10__3,
        axis=1,
        reduction="none",
    )
    c_lifted_tensor_0__8 = op.Constant(
        value=from_array(np.array([1.0], dtype=np.float32), name="value")
    )
    c_lifted_tensor_1__8 = op.Constant(
        value=from_array(np.array(3, dtype=np.int64), name="value")
    )
    c_lifted_tensor_2__8 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value")
    )
    init7_s_1__8 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value")
    )
    init7_s1__1__8 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value")
    )
    init7_s_0__8 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value")
    )
    init11_s___8 = op.Constant(
        value=from_array(np.array(1.0, dtype=np.float64), name="value")
    )
    init1_s___8 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value")
    )
    init7_s1_0__8 = op.Constant(
        value=from_array(np.array([0], dtype=np.int64), name="value")
    )
    select__8 = op.Gather(isnan, init7_s_1__8, axis=1)
    index__8 = op.Gather(select__8, nonzero_numpy__0, axis=0)
    select_1__8 = op.Gather(logical_not, init7_s_1__8, axis=1)
    _onx_nonzero_select_10__8 = op.NonZero(select_1__8)
    nonzero_numpy__0__8 = op.Reshape(_onx_nonzero_select_10__8, init7_s1__1__8)
    _shape_getitem_20__8 = op.Shape(nonzero_numpy__0__8, end=1, start=0)
    sym_size_int_20__8 = op.SqueezeAnyOpset(_shape_getitem_20__8)
    view__8_18 = op.Reshape(index__8, init7_s1__1__8)
    _onx_nonzero_view0__8 = op.NonZero(view__8_18)
    nonzero_numpy_1__0__8 = op.Reshape(_onx_nonzero_view0__8, init7_s1__1__8)
    index_1__8 = op.Gather(nonzero_numpy__0, nonzero_numpy_1__0__8, axis=0)
    index_2__8 = op.Gather(c_torch_knnimputer__make_dict_idx_map, index_1__8, axis=0)
    index_3__8 = op.Gather(c_torch_knnimputer_dist, index_2__8, axis=0)
    _onx_gather_index_30__8 = op.Gather(index_3__8, nonzero_numpy__0__8, axis=1)
    isnan__8_19 = op.IsNaN(_onx_gather_index_30__8)
    _onx_cast_isnan0__8 = op.Cast(isnan__8_19, to=6)
    _onx_reducemin_cast_isnan00__8 = op.ReduceMinAnyOpset(
        _onx_cast_isnan0__8, c_lifted_tensor_2__8, keepdims=0
    )
    all_1__8 = op.Cast(_onx_reducemin_cast_isnan00__8, to=9)
    index_5__8 = op.Compress(index_1__8, all_1__8, axis=0)
    select_2__8 = op.Gather(_mask_fit_x, init7_s_1__8, axis=1)
    bitwise_not__8_20 = op.Not(select_2__8)
    _to_copy__8_21 = op.Cast(bitwise_not__8_20, to=11)
    _to_copy_1__8 = op.Cast(_to_copy__8_21, to=1)
    sum_1__8 = op.ReduceSumAnyOpset(_to_copy_1__8, keepdims=0)
    _reshape_init11_s_0__8 = op.Reshape(init11_s___8, c_lifted_tensor_2__8)
    eq_23__8 = op.Equal(_to_copy__8_21, _reshape_init11_s_0__8)
    select_3__8 = op.Gather(_fit_x, init7_s_1__8, axis=1)
    index_6__8 = op.Compress(select_3__8, eq_23__8, axis=0)
    sum_2__8 = op.ReduceSumAnyOpset(index_6__8, keepdims=0)
    _to_copy_2__8 = op.Cast(sum_2__8, to=1)
    gt__8 = op.Greater(sum_1__8, init1_s___8)
    where__8 = op.Where(gt__8, sum_1__8, c_lifted_tensor_0__8)
    _reshape__to_copy_20__8 = op.Reshape(_to_copy_2__8, c_lifted_tensor_2__8)
    div__8 = op.Div(_reshape__to_copy_20__8, where__8)
    select_4__8 = op.Gather(c_torch_knnimputer_columns_0_, init7_s_1__8, axis=1)
    view_1__8 = op.SqueezeAnyOpset(div__8, init7_s1_0__8)
    _onx_unsqueeze_index_50__8 = op.UnsqueezeAnyOpset(index_5__8, init7_s1__1__8)
    _shape_index_502__8 = op.Shape(index_5__8)
    _onx_expand_view_10__8 = op.Expand(view_1__8, _shape_index_502__8)
    index_put__8 = op.ScatterND(
        select_4__8, _onx_unsqueeze_index_50__8, _onx_expand_view_10__8
    )
    _onx_unsqueeze_index_put0__8 = op.UnsqueezeAnyOpset(index_put__8, init7_s_1__8)
    _shape_unsqueeze_index_put00__8 = op.Shape(_onx_unsqueeze_index_put0__8)
    _onx_expand_c_lifted_tensor_20__8 = op.Expand(
        c_lifted_tensor_2__8, _shape_unsqueeze_index_put00__8
    )
    select_scatter__8_22 = op.ScatterElements(
        c_torch_knnimputer_columns_0_,
        _onx_expand_c_lifted_tensor_20__8,
        _onx_unsqueeze_index_put0__8,
        axis=1,
        reduction="none",
    )
    bitwise_not_1__8 = op.Not(all_1__8)
    index_7__8 = op.Compress(index_1__8, bitwise_not_1__8, axis=0)
    index_8__8 = op.Gather(c_torch_knnimputer__make_dict_idx_map, index_7__8, axis=0)
    index_9__8 = op.Gather(c_torch_knnimputer_dist, index_8__8, axis=0)
    _onx_gather_index_90__8 = op.Gather(index_9__8, nonzero_numpy__0__8, axis=1)
    lt__8 = op.Less(c_lifted_tensor_1__8, sym_size_int_20__8)
    where_1__8 = op.Where(lt__8, c_lifted_tensor_1__8, sym_size_int_20__8)
    le_3__8 = op.LessOrEqual(where_1__8, init7_s_0__8)
    where_2__8 = op.Where(le_3__8, c_lifted_tensor_2__8, where_1__8)
    select_6__8 = op.Gather(_fit_x, init7_s_1__8, axis=1)
    index_11__8 = op.Gather(select_6__8, nonzero_numpy__0__8, axis=0)
    select_7__8 = op.Gather(_mask_fit_x, init7_s_1__8, axis=1)
    index_12__8 = op.Gather(select_7__8, nonzero_numpy__0__8, axis=0)
    c_lifted_tensor_0__9 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value")
    )
    c_lifted_tensor_1__9 = op.Constant(
        value=from_array(np.array([1.0], dtype=np.float64), name="value")
    )
    init7_s1__1__9 = op.Constant(
        value=from_array(np.array([-1], dtype=np.int64), name="value")
    )
    init11_s___9 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float64), name="value")
    )
    init7_s_0__10 = op.Constant(
        value=from_array(np.array(0, dtype=np.int64), name="value")
    )
    init7_s1_1__10 = op.Constant(
        value=from_array(np.array([1], dtype=np.int64), name="value")
    )
    init7_s_1__10 = op.Constant(
        value=from_array(np.array(1, dtype=np.int64), name="value")
    )
    _shape_dist_pot_donors0__10 = op.Shape(_onx_gather_index_90__8, end=1, start=0)
    sym_size_int_4__10 = op.SqueezeAnyOpset(_shape_dist_pot_donors0__10)
    (
        unused_topk_values__10,
        c_torch_knnimputer_columns_1___calc_impute__donors_idx__0__9,
    ) = op.TopK(_onx_gather_index_90__8, where_2__8, largest=0, sorted=1)
    arange__10_25 = op.Range(init7_s_0__10, sym_size_int_4__10, init7_s_1__10)
    unsqueeze__10_26 = op.UnsqueezeAnyOpset(arange__10_25, init7_s1_1__10)
    _onx_gathernd_dist_pot_donors0__10 = op.GatherND(
        _onx_gather_index_90__8, unsqueeze__10_26, batch_dims=0
    )
    c_torch_knnimputer_columns_1___calc_impute__donors_idx__1__9 = op.GatherElements(
        _onx_gathernd_dist_pot_donors0__10,
        c_torch_knnimputer_columns_1___calc_impute__donors_idx__0__9,
        axis=1,
    )
    c_lifted_tensor_0__11 = op.Constant(
        value=from_array(np.array(0.0, dtype=np.float32), name="value")
    )
    _shape_donors_dist0__11 = op.Shape(
        c_torch_knnimputer_columns_1___calc_impute__donors_idx__1__9
    )
    ones_like__11_27 = op.ConstantOfShape(
        _shape_donors_dist0__11,
        value=from_array(np.array([1.0], dtype=np.float32), name="value"),
    )
    isnan__11_28 = op.IsNaN(
        c_torch_knnimputer_columns_1___calc_impute__donors_idx__1__9
    )
    c_torch_knnimputer_columns_1___calc_impute__weights__9 = op.Where(
        isnan__11_28, c_lifted_tensor_0__11, ones_like__11_27
    )
    _reshape_fit_x_col0__9 = op.Reshape(index_11__8, init7_s1__1__9)
    take__9_23 = op.Gather(
        _reshape_fit_x_col0__9,
        c_torch_knnimputer_columns_1___calc_impute__donors_idx__0__9,
    )
    _reshape_mask_fit_x_col0__9 = op.Reshape(index_12__8, init7_s1__1__9)
    take_1__9 = op.Gather(
        _reshape_mask_fit_x_col0__9,
        c_torch_knnimputer_columns_1___calc_impute__donors_idx__0__9,
    )
    _to_copy__9_24 = op.Cast(take_1__9, to=7)
    sub_12__9 = op.Sub(c_lifted_tensor_0__9, _to_copy__9_24)
    _to_copy__12_29 = op.Cast(sub_12__9, to=11)
    _to_copy_1__12 = op.Cast(
        c_torch_knnimputer_columns_1___calc_impute__weights__9, to=11
    )
    c_torch_knnimputer_columns_1___calc_impute__make_new_neights__9 = op.Mul(
        _to_copy__12_29, _to_copy_1__12
    )
    sum_1__9 = op.ReduceSumAnyOpset(
        c_torch_knnimputer_columns_1___calc_impute__make_new_neights__9,
        c_lifted_tensor_0__9,
        keepdims=1,
    )
    _reshape_init11_s_0__9 = op.Reshape(init11_s___9, c_lifted_tensor_0__9)
    eq_17__9 = op.Equal(sum_1__9, _reshape_init11_s_0__9)
    where__9 = op.Where(eq_17__9, c_lifted_tensor_1__9, sum_1__9)
    mul_17__9 = op.Mul(
        take__9_23, c_torch_knnimputer_columns_1___calc_impute__make_new_neights__9
    )
    sum_2__9 = op.ReduceSumAnyOpset(mul_17__9, c_lifted_tensor_0__9, keepdims=1)
    div__9 = op.Div(sum_2__9, where__9)
    _onx_squeeze_div0__9 = op.SqueezeAnyOpset(div__9, c_lifted_tensor_0__9)
    c_torch_knnimputer_columns_1___calc_impute__8 = op.Cast(_onx_squeeze_div0__9, to=1)
    select_9__8 = op.Gather(select_scatter__8_22, init7_s_1__8, axis=1)
    _onx_unsqueeze_index_70__8 = op.UnsqueezeAnyOpset(index_7__8, init7_s1__1__8)
    index_put_1__8 = op.ScatterND(
        select_9__8,
        _onx_unsqueeze_index_70__8,
        c_torch_knnimputer_columns_1___calc_impute__8,
    )
    _onx_unsqueeze_index_put_10__8 = op.UnsqueezeAnyOpset(index_put_1__8, init7_s_1__8)
    _shape_unsqueeze_index_put_100__8 = op.Shape(_onx_unsqueeze_index_put_10__8)
    _onx_expand_c_lifted_tensor_202__8 = op.Expand(
        c_lifted_tensor_2__8, _shape_unsqueeze_index_put_100__8
    )
    c_torch_knnimputer_columns_1_ = op.ScatterElements(
        select_scatter__8_22,
        _onx_expand_c_lifted_tensor_202__8,
        _onx_unsqueeze_index_put_10__8,
        axis=1,
        reduction="none",
    )
    return op.Compress(c_torch_knnimputer_columns_1_, _valid_mask, axis=1)


def convert_knn_imputer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *KNNImputer* into *ONNX*.
    """
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
        operator.inputs[0].full_name,
    )
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
