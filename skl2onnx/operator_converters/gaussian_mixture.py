# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from scipy.special import digamma
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
try:
    from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky
except ImportError:
    # scikit-learn < 0.22
    from sklearn.mixture.gaussian_mixture import _compute_log_det_cholesky
from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxSub, OnnxMul, OnnxGemm, OnnxReduceSumSquare,
    OnnxReduceLogSumExp, OnnxExp, OnnxArgMax, OnnxConcat,
    OnnxReduceSum, OnnxLog, OnnxReduceMax, OnnxEqual, OnnxCast
)
from ..proto import onnx_proto


def _estimate_log_gaussian_prob(X, means, precisions_chol,
                                covariance_type, dtype, op_version,
                                combined_reducesum):
    """
    Converts the same function into ONNX.
    Returns log probabilities.
    """
    n_components = means.shape[0]
    n_features = means.shape[1]
    opv = op_version

    # self._estimate_log_prob(X)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features).astype(
            dtype)

    if covariance_type == 'full':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_components, n_features, n_features)

        # log_prob = np.empty((n_samples, n_components))
        # for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        #     y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        #     log_prob[:, k] = np.sum(np.square(y), axis=1)

        ys = []
        for c in range(n_components):
            prec_chol = precisions_chol[c, :, :]
            cst = - np.dot(means[c, :], prec_chol)
            y = OnnxGemm(X, prec_chol.astype(dtype),
                         cst.astype(dtype), alpha=1.,
                         beta=1., op_version=opv)
            if combined_reducesum:
                y2s = OnnxReduceSum(OnnxMul(y, y, op_version=opv),
                                    axes=[1], op_version=opv)
            else:
                y2s = OnnxReduceSumSquare(y, axes=[1], op_version=opv)
            ys.append(y2s)
        log_prob = OnnxConcat(*ys, axis=1, op_version=opv)

    elif covariance_type == 'tied':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_features, n_features)

        # log_prob = np.empty((n_samples, n_components))
        # for k, mu in enumerate(means):
        #     y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
        #     log_prob[:, k] = np.sum(np.square(y), axis=1)

        ys = []
        for f in range(n_components):
            cst = - np.dot(means[f, :], precisions_chol)
            y = OnnxGemm(X, precisions_chol.astype(dtype),
                         cst.astype(dtype),
                         alpha=1., beta=1., op_version=opv)
            if combined_reducesum:
                y2s = OnnxReduceSum(OnnxMul(y, y, op_version=opv),
                                    axes=[1], op_version=opv)
            else:
                y2s = OnnxReduceSumSquare(y, axes=[1], op_version=opv)
            ys.append(y2s)
        log_prob = OnnxConcat(*ys, axis=1, op_version=opv)

    elif covariance_type == 'diag':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_components, n_features)

        # precisions = precisions_chol ** 2
        # log_prob = (np.sum((means ** 2 * precisions), 1) -
        #             2. * np.dot(X, (means * precisions).T) +
        #             np.dot(X ** 2, precisions.T))

        precisions = (precisions_chol ** 2)
        mp = np.sum((means ** 2 * precisions), 1)
        zeros = np.zeros((n_components, ), dtype=dtype)
        xmp = OnnxGemm(
            X, (means * precisions).T.astype(dtype),
            zeros, alpha=-2., beta=0., op_version=opv)
        term = OnnxGemm(OnnxMul(X, X, op_version=opv),
                        precisions.T.astype(dtype),
                        zeros, alpha=1., beta=0., op_version=opv)
        log_prob = OnnxAdd(
            OnnxAdd(mp.astype(dtype), xmp, op_version=opv),
            term, op_version=opv)

    elif covariance_type == 'spherical':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) = (n_components, )

        # precisions = precisions_chol ** 2
        # log_prob = (np.sum(means ** 2, 1) * precisions -
        #             2 * np.dot(X, means.T * precisions) +
        #             np.outer(row_norms(X, squared=True), precisions))

        zeros = np.zeros((n_components, ), dtype=dtype)
        precisions = (precisions_chol ** 2).astype(dtype)
        if combined_reducesum:
            normX = OnnxReduceSum(OnnxMul(X, X, op_version=opv),
                                  axes=[1], op_version=opv)
        else:
            normX = OnnxReduceSumSquare(X, axes=[1], op_version=opv)
        outer = OnnxGemm(
            normX, precisions[np.newaxis, :].astype(dtype),
            zeros.astype(dtype), alpha=1., beta=1., op_version=opv)
        xmp = OnnxGemm(
            X, (means.T * precisions).astype(dtype),
            zeros, alpha=-2., beta=0., op_version=opv)
        mp = (np.sum(means ** 2, 1) * precisions).astype(dtype)
        log_prob = OnnxAdd(mp, OnnxAdd(xmp, outer, op_version=opv),
                           op_version=opv)
    else:
        raise RuntimeError("Unknown op.covariance_type='{}'. Upgrade "
                           "to a more recent version of skearn-onnx "
                           "or raise an issue.".format(covariance_type))
    # -.5 * (cst + log_prob) + log_det
    cst = np.array([n_features * np.log(2 * np.pi)])
    add = OnnxAdd(cst.astype(dtype), log_prob, op_version=opv)
    mul = OnnxMul(add, np.array([-0.5], dtype=dtype),
                  op_version=opv)
    if isinstance(log_det, float):
        log_det = np.array([log_det], dtype=dtype)
    weighted_log_prob = OnnxAdd(
        OnnxAdd(mul, log_det, op_version=opv),
        log_weights.astype(dtype), op_version=opv)

    # labels
    if container.is_allowed('ArgMax'):
        labels = OnnxArgMax(weighted_log_prob, axis=1,
                            output_names=out[:1], op_version=opv)
    else:
        mxlabels = OnnxReduceMax(weighted_log_prob, axes=[1], op_version=opv)
        zeros = OnnxEqual(
            OnnxSub(weighted_log_prob, mxlabels, op_version=opv),
            np.array([0], dtype=dtype),
            op_version=opv)
        toint = OnnxCast(zeros, to=onnx_proto.TensorProto.INT64,
                         op_version=opv)
        mulind = OnnxMul(toint,
                         np.arange(n_components).astype(np.int64),
                         op_version=opv)
        labels = OnnxReduceMax(mulind, axes=[1], output_names=out[:1],
                               op_version=opv)

    # def _estimate_log_prob_resp():
    # np.exp(log_resp)
    # weighted_log_prob = self._estimate_weighted_log_prob(X)
    # log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    # with np.errstate(under='ignore'):
    #    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    if add_score:
        outnames = out[2:3]
    else:
        outnames = None

    if combined_reducesum:
        max_weight = OnnxReduceMax(weighted_log_prob, axes=[1], op_version=opv)
        log_prob_norm_demax = OnnxLog(
            OnnxReduceSum(
                OnnxExp(
                    OnnxSub(weighted_log_prob, max_weight, op_version=opv),
                    op_version=opv),
                axes=[1], op_version=opv),
            op_version=opv)
        log_prob_norm = OnnxAdd(log_prob_norm_demax, max_weight,
                                op_version=opv, output_names=out[2:3])
    else:
        log_prob_norm = OnnxReduceLogSumExp(
            weighted_log_prob, axes=[1], op_version=opv,
            output_names=outnames)
    log_resp = OnnxSub(weighted_log_prob, log_prob_norm, op_version=opv)

    # probabilities
    probs = OnnxExp(log_resp, output_names=out[1:2], op_version=opv)

    # final
    labels.add_to(scope, container)
    probs.add_to(scope, container)
    if add_score:
        log_prob_norm.add_to(scope, container)


register_converter('SklearnGaussianMixture', convert_sklearn_gaussian_mixture,
                   options={'score_samples': [True, False]})
register_converter('SklearnBayesianGaussianMixture',
                   convert_sklearn_gaussian_mixture,
                   options={'score_samples': [True, False]})
