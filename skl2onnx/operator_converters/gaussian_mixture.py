# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
try:
    from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky
except ImportError:
    # scikit-learn < 0.22
    from sklearn.mixture.gaussian_mixture import _compute_log_det_cholesky
from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxSub, OnnxMul, OnnxGemm, OnnxReduceSumSquare,
    OnnxReduceLogSumExp, OnnxExp, OnnxArgMax, OnnxConcat
)


def convert_sklearn_gaussian_mixture(scope, operator, container):
    """
    Converter for *GaussianMixture*,
    *BayesianGaussianMixture*.
    Parameters which change the prediction function:

    * *covariance_type*
    """
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    n_features = X.type.shape[1]
    n_components = op.means_.shape[0]
    opv = container.target_opset

    # All comments come from scikit-learn code and tells
    # which functions is being onnxified.
    # def _estimate_weighted_log_prob(self, X):
    # self._estimate_log_prob(X) + self._estimate_log_weights()
    log_weights = np.log(op.weights_)  # self._estimate_log_weights()

    # self._estimate_log_prob(X)
    log_det = _compute_log_det_cholesky(
        op.precisions_cholesky_, op.covariance_type, n_features)

    if op.covariance_type == 'full':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_components, n_features, n_features)

        # log_prob = np.empty((n_samples, n_components))
        # for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        #     y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        #     log_prob[:, k] = np.sum(np.square(y), axis=1)

        ys = []
        for c in range(n_components):
            prec_chol = op.precisions_cholesky_[c, :, :]
            cst = - np.dot(op.means_[c, :], prec_chol)
            y = OnnxGemm(X, prec_chol, cst, alpha=1., beta=1., op_version=opv)
            y2s = OnnxReduceSumSquare(y, axes=[1], op_version=opv)
            ys.append(y2s)
        log_prob = OnnxConcat(*ys, axis=1, op_version=opv)

    elif op.covariance_type == 'tied':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_features, n_features)

        # log_prob = np.empty((n_samples, n_components))
        # for k, mu in enumerate(means):
        #     y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
        #     log_prob[:, k] = np.sum(np.square(y), axis=1)

        precisions_chol = op.precisions_cholesky_
        ys = []
        for f in range(n_components):
            cst = - np.dot(op.means_[f, :], precisions_chol)
            y = OnnxGemm(X, precisions_chol, cst,
                         alpha=1., beta=1., op_version=opv)
            y2s = OnnxReduceSumSquare(y, axes=[1], op_version=opv)
            ys.append(y2s)
        log_prob = OnnxConcat(*ys, axis=1, op_version=opv)

    elif op.covariance_type == 'diag':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_components, n_features)

        # precisions = precisions_chol ** 2
        # log_prob = (np.sum((means ** 2 * precisions), 1) -
        #             2. * np.dot(X, (means * precisions).T) +
        #             np.dot(X ** 2, precisions.T))

        precisions = op.precisions_cholesky_ ** 2
        mp = np.sum((op.means_ ** 2 * precisions), 1)
        zeros = np.zeros((n_components, ))
        xmp = OnnxGemm(X, (op.means_ * precisions).T, zeros,
                       alpha=-2., beta=0., op_version=opv)
        term = OnnxGemm(OnnxMul(X, X, op_version=opv),
                        precisions.T, zeros, alpha=1., beta=0.,
                        op_version=opv)
        log_prob = OnnxAdd(OnnxAdd(mp, xmp, op_version=opv),
                           term, op_version=opv)

    elif op.covariance_type == 'spherical':
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) = (n_components, )

        # precisions = precisions_chol ** 2
        # log_prob = (np.sum(means ** 2, 1) * precisions -
        #             2 * np.dot(X, means.T * precisions) +
        #             np.outer(row_norms(X, squared=True), precisions))

        zeros = np.zeros((n_components, ))
        precisions = op.precisions_cholesky_ ** 2
        normX = OnnxReduceSumSquare(X, axes=[1], op_version=opv)
        outer = OnnxGemm(normX, precisions[np.newaxis, :], zeros,
                         alpha=1., beta=1., op_version=opv)
        xmp = OnnxGemm(X, (op.means_.T * precisions), zeros,
                       alpha=-2., beta=0., op_version=opv)
        mp = np.sum(op.means_ ** 2, 1) * precisions
        log_prob = OnnxAdd(mp, OnnxAdd(xmp, outer, op_version=opv),
                           op_version=opv)
    else:
        raise RuntimeError("Unknown op.covariance_type='{}'. Upgrade "
                           "to a mroe recent version of skearn-onnx "
                           "or raise an issue.".format(op.covariance_type))

    # -.5 * (cst + log_prob) + log_det
    cst = np.array([n_features * np.log(2 * np.pi)])
    add = OnnxAdd(cst, log_prob, op_version=opv)
    mul = OnnxMul(add, np.array([-0.5]), op_version=opv)
    if isinstance(log_det, float):
        log_det = np.array([log_det])
    weighted_log_prob = OnnxAdd(OnnxAdd(mul, log_det, op_version=opv),
                                log_weights, op_version=opv)

    # labels
    labels = OnnxArgMax(weighted_log_prob, axis=1,
                        output_names=out[:1], op_version=opv)

    # def _estimate_log_prob_resp():
    # np.exp(log_resp)
    # weighted_log_prob = self._estimate_weighted_log_prob(X)
    # log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    # with np.errstate(under='ignore'):
    #    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

    log_prob_norm = OnnxReduceLogSumExp(
        weighted_log_prob, axes=[1], op_version=opv)
    log_resp = OnnxSub(weighted_log_prob, log_prob_norm, op_version=opv)

    # probabilities
    probs = OnnxExp(log_resp, output_names=out[1:], op_version=opv)

    # final
    labels.add_to(scope, container)
    probs.add_to(scope, container)


register_converter('SklearnGaussianMixture', convert_sklearn_gaussian_mixture)
register_converter('SklearnBayesianGaussianMixture',
                   convert_sklearn_gaussian_mixture)
