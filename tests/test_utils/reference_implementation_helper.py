# SPDX-License-Identifier: Apache-2.0
import numpy as np


def ErfInv(x):
    sgn = -1.0 if x < 0 else 1.0
    x = (1.0 - x) * (1 + x)
    log = np.log(x)
    v = 2.0 / (3.14159 * 0.147) + 0.5 * log
    v2 = 1.0 / 0.147 * log
    v3 = -v + np.sqrt(v * v - v2)
    x = sgn * np.sqrt(v3)
    return x


def ComputeLogistic(val):
    v = 1.0 / (1.0 + np.exp(-np.abs(val)))
    return (1.0 - v) if val < 0 else v


def ComputeProbit(val):
    return 1.41421356 * ErfInv(val * 2 - 1)


def ComputeSoftmax(values):
    v_max = values.max()
    values[:] = np.exp(values - v_max)
    this_sum = values.sum()
    values /= this_sum
    return values


def ComputeSoftmaxZero(values):
    v_max = values.max()
    exp_neg_v_max = np.exp(-v_max)
    s = 0
    for i in range(len(values)):
        v = values[i]
        if v > 0.0000001 or v < -0.0000001:
            values[i] = np.exp(v - v_max)
            s += values[i]
        else:
            values[i] *= exp_neg_v_max
    values[i] /= s
    return values


def sigmoid_probability(score, proba, probb):
    # ref: https://github.com/arnaudsj/libsvm/blob/
    # eaaefac5ebd32d0e07902e1ae740e038eaaf0826/svm.cpp#L1818
    val = score * proba + probb
    return 1 - ComputeLogistic(val)


def multiclass_probability(k, R):
    max_iter = max(100, k)
    Q = np.empty((k, k), dtype=R.dtype)
    Qp = np.empty((k,), dtype=R.dtype)
    P = np.empty((k,), dtype=R.dtype)
    eps = 0.005 / k

    for t in range(0, k):
        P[t] = 1.0 / k
        Q[t, t] = 0
        for j in range(t):
            Q[t, t] += R[j, t] * R[j, t]
            Q[t, j] = Q[j, t]
        for j in range(t + 1, k):
            Q[t, t] += R[j, t] * R[j, t]
            Q[t, j] = -R[j, t] * R[t, j]

    for _it in range(max_iter):
        # stopping condition, recalculate QP,pQP for numerical accuracy
        pQp = 0
        for t in range(0, k):
            Qp[t] = 0
            for j in range(k):
                Qp[t] += Q[t, j] * P[j]
            pQp += P[t] * Qp[t]

        max_error = 0
        for t in range(0, k):
            error = np.abs(Qp[t] - pQp)
            if error > max_error:
                max_error = error
        if max_error < eps:
            break

        for t in range(k):
            diff = (-Qp[t] + pQp) / Q[t, t]
            P[t] += diff
            pQp = (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff) ** 2
            for j in range(k):
                Qp[j] = (Qp[j] + diff * Q[t, j]) / (1 + diff)
                P[j] /= 1 + diff
    return P


def write_scores(n_classes, scores, post_transform, add_second_class):
    if n_classes >= 2:
        if post_transform == "PROBIT":
            res = []
            for score in scores:
                res.append(ComputeProbit(score))
            return np.array(res, dtype=scores.dtype)
        if post_transform == "LOGISTIC":
            res = []
            for score in scores:
                res.append(ComputeLogistic(score))
            return np.array(res, dtype=scores.dtype)
        if post_transform == "SOFTMAX":
            return ComputeSoftmax(scores)
        if post_transform == "SOFTMAX_ZERO":
            return ComputeSoftmaxZero(scores)
        return scores
    if n_classes == 1:
        if post_transform == "PROBIT":
            return np.array([ComputeProbit(scores[0])], dtype=scores.dtype)
        if add_second_class == 0:
            res = np.array([1 - scores[0], scores[0]], dtype=scores.dtype)
        elif add_second_class == 1:
            res = np.array([1 - scores[0], scores[0]], dtype=scores.dtype)
        elif add_second_class in (2, 3):
            if post_transform == "LOGISTIC":
                return np.array(
                    [ComputeLogistic(-scores[0]), ComputeLogistic(scores[0])],
                    dtype=scores.dtype,
                )
            return np.array([-scores[0], scores[0]], dtype=scores.dtype)
        return np.array([scores[0]], dtype=scores.dtype)
    raise NotImplementedError(f"n_classes={n_classes} not supported.")


def set_score_svm(
    max_weight,
    maxclass,
    n,
    post_transform,
    has_proba,
    weights_are_all_positive_,
    classlabels,
    posclass,
    negclass,
):
    write_additional_scores = -1
    if len(classlabels) == 2:
        write_additional_scores = 2 if post_transform == "NONE" else 0
        if not has_proba:
            if weights_are_all_positive_ and max_weight >= 0.5:
                return classlabels[1], write_additional_scores
            if max_weight > 0 and not weights_are_all_positive_:
                return classlabels[1], write_additional_scores
        return classlabels[maxclass], write_additional_scores
    if max_weight > 0:
        return posclass, write_additional_scores
    return negclass, write_additional_scores
