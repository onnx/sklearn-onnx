# SPDX-License-Identifier: Apache-2.0
"""
Helpers to test runtimes.
"""

import numpy as np
from onnx import numpy_helper
from onnx.defs import onnx_opset_version


def _to_str(s):
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return s


def _attribute_value(attr):
    if attr.HasField("f"):
        return attr.f
    if attr.HasField("i"):
        return attr.i
    if attr.HasField("s"):
        return _to_str(attr.s)
    if attr.HasField("t"):
        return numpy_helper.to_array(attr.t)
    if attr.floats:
        return list(attr.floats)
    if attr.ints:
        return list(attr.ints)
    if attr.strings:
        return list(map(_to_str, attr.strings))
    raise NotImplementedError("Unable to return a value for attribute %r." % attr)


class SVMAttributes:
    def __init__(self):
        self._names = []

    def add(self, name, value):
        if isinstance(value, list) and name not in {"kernel_params"}:
            if name in {"vectors_per_class"}:
                value = np.array(value, dtype=np.int64)
            else:
                value = np.array(value, dtype=np.float32)
        setattr(self, name, value)

    def __str__(self):
        rows = ["Attributes"]
        for name in self._names:
            rows.append(f"  {name}={getattr(self, name)}")
        return "\n".join(rows)


class SVMCommon:
    """
    Base class for SVM.
    """

    def __init__(self, **kwargs):
        self.atts = SVMAttributes()

        for name, value in kwargs.items():
            self.atts.add(name, value)

        if self.atts.kernel_params:
            self.gamma_ = self.atts.kernel_params[0]
            self.coef0_ = self.atts.kernel_params[1]
            self.degree_ = int(self.atts.kernel_params[2])
        else:
            self.gamma_ = 0.0
            self.coef0_ = 0.0
            self.degree_ = 0

    def __str__(self):
        rows = ["TreeEnsemble", f"root_index={self.root_index}", str(self.atts)]
        return "\n".join(rows)

    def kernel_dot(self, pA, pB, kernel):
        k = kernel.lower()
        if k == "poly":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return s**self.degree_
        if k == "sigmoid":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return np.tanh(s)
        if k == "rbf":
            diff = pA - pB
            s = (diff * diff).sum()
            return np.exp(-self.gamma_ * s)
        if k == "linear":
            return np.dot(pA, pB)
        raise ValueError(f"Unexpected kernel={kernel!r}.")

    def run(self, X):
        if self.atts.n_supports > 0:
            # length of each support vector
            mode_ = "SVM_SVC"
            kernel_type_ = self.atts.kernel_type
            sv = self.atts.support_vectors.reshape((self.atts.n_supports, -1))
        else:
            mode_ = "SVM_LINEAR"
            kernel_type_ = "LINEAR"

        z = np.empty((X.shape[0], 1), dtype=X.dtype)
        for n in range(X.shape[0]):
            s = 0.0

            if mode_ == "SVM_SVC":
                for j in range(self.atts.n_supports):
                    d = self.kernel_dot(X[n], sv[j], kernel_type_)
                    s += self.atts.coefficients[j] * d
                s += self.atts.rho[0]
            elif mode_ == "SVM_LINEAR":
                s = self.kernel_dot(X, self.atts.coefficients, kernel_type_)
                s += self.atts.rho[0]

            if self.atts.one_class:
                z[n, 0] = 1 if s > 0 else -1
            else:
                z[n, 0] = s
        return z


if onnx_opset_version() >= 18:
    from onnx.reference.op_run import OpRun

    try:
        from .reference_implementation_helper import (
            write_scores,
            set_score_svm,
            multiclass_probability,
            sigmoid_probability,
        )
    except ImportError:
        from reference_implementation_helper import (
            write_scores,
            set_score_svm,
            multiclass_probability,
            sigmoid_probability,
        )

    class SVMRegressor(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(
            self,
            X,
            coefficients=None,
            kernel_params=None,
            kernel_type=None,
            n_targets=None,
            n_supports=None,
            one_class=None,
            post_transform=None,
            rho=None,
            support_vectors=None,
        ):
            svm = SVMCommon(
                coefficients=coefficients,
                kernel_params=kernel_params,
                kernel_type=kernel_type,
                n_targets=n_targets,
                n_supports=n_supports,
                one_class=one_class,
                post_transform=post_transform,
                rho=rho,
                support_vectors=support_vectors,
            )
            self._svm = svm
            res = svm.run(X)

            if post_transform in (None, "NONE"):
                return (res,)
            raise NotImplementedError(
                f"post_transform={post_transform!r} not implemented."
            )

    class SVMClassifier(OpRun):
        op_domain = "ai.onnx.ml"

        def _run_linear(self, X, coefs, class_count_, kernel_type_):
            scores = []
            for j in range(class_count_):
                d = self._svm.kernel_dot(X, coefs[j], kernel_type_)
                score = self._svm.rho[0] + d
                scores.append(score)
            return np.array(scores, dtype=X.dtype)

        def _run_svm(
            self,
            X,
            sv,
            vector_count_,
            kernel_type_,
            class_count_,
            starting_vector_,
            coefs,
        ):
            evals = 0

            kernels = []
            for j in range(vector_count_):
                kernels.append(self._svm.kernel_dot(X, sv[j], kernel_type_))
            kernels = np.array(kernels)

            votes = np.zeros((class_count_,), dtype=X.dtype)
            scores = []
            for i in range(class_count_):
                si_i = starting_vector_[i]
                class_i_sc = self._svm.atts.vectors_per_class[i]

                for j in range(i + 1, class_count_):
                    si_j = starting_vector_[j]
                    class_j_sc = self._svm.atts.vectors_per_class[j]

                    s1 = np.dot(
                        coefs[j - 1, si_i : si_i + class_i_sc],
                        kernels[si_i : si_i + class_i_sc],
                    )
                    s2 = np.dot(
                        coefs[i, si_j : si_j + class_j_sc],
                        kernels[si_j : si_j + class_j_sc],
                    )

                    s = self._svm.atts.rho[evals] + s1 + s2
                    scores.append(s)
                    if s > 0:
                        votes[i] += 1
                    else:
                        votes[j] += 1
                    evals += 1
            return votes, np.array(scores, dtype=X.dtype)

        def _probabilities(self, scores, class_count_):
            probsp2 = np.zeros((class_count_, class_count_), dtype=scores.dtype)

            index = 0
            for i in range(class_count_):
                for j in range(i + 1, class_count_):
                    val1 = sigmoid_probability(
                        scores[index],
                        self._svm.atts.prob_a[index],
                        self._svm.atts.prob_b[index],
                    )
                    val2 = max(val1, 1.0e-7)
                    val2 = min(val2, (1 - 1.0e-7))
                    probsp2[i, j] = val2
                    probsp2[j, i] = 1 - val2
                    index += 1
            return multiclass_probability(class_count_, probsp2)

        def _compute_final_scores(
            self, votes, scores, weights_are_all_positive_, has_proba, classlabels_ints
        ):
            max_weight = 0
            if len(votes):
                max_class = np.argmax(votes)
                max_weight = votes[max_class]
            else:
                max_class = np.argmax(scores)
                max_weight = scores[max_class]

            write_additional_scores = -1
            if self._svm.atts.rho.size == 1:
                label, write_additional_scores = set_score_svm(
                    max_weight,
                    max_class,
                    0,
                    self._svm.atts.post_transform,
                    has_proba,
                    weights_are_all_positive_,
                    classlabels_ints,
                    1,
                    0,
                )
            elif classlabels_ints is not None and len(classlabels_ints) > 0:
                label = classlabels_ints[max_class]
            else:
                label = max_class

            new_scores = write_scores(
                scores.size,
                scores,
                self._svm.atts.post_transform,
                write_additional_scores,
            )
            return label, new_scores

        def _run(
            self,
            X,
            classlabels_ints=None,
            classlabels_strings=None,
            coefficients=None,
            kernel_params=None,
            kernel_type=None,
            post_transform=None,
            prob_a=None,
            prob_b=None,
            rho=None,
            support_vectors=None,
            vectors_per_class=None,
        ):
            svm = SVMCommon(
                coefficients=coefficients,
                kernel_params=kernel_params,
                kernel_type=kernel_type,
                post_transform=post_transform,
                prob_a=prob_a,
                prob_b=prob_b,
                rho=rho,
                support_vectors=support_vectors,
                vectors_per_class=vectors_per_class,
            )
            self._svm = svm

            vector_count_ = 0
            class_count_ = 0
            starting_vector_ = []
            for vc in svm.atts.vectors_per_class:
                starting_vector_.append(vector_count_)
                vector_count_ += vc

            class_count_ = max(len(classlabels_ints or classlabels_strings or []), 1)
            if vector_count_ > 0:
                # length of each support vector
                mode_ = "SVM_SVC"
                sv = svm.atts.support_vectors.reshape((vector_count_, -1))
                kernel_type_ = svm.atts.kernel_type
                coefs = svm.atts.coefficients.reshape((-1, vector_count_))
            else:
                # liblinear mode
                mode_ = "SVM_LINEAR"
                kernel_type_ = "LINEAR"
                coefs = svm.atts.coefficients.reshape((class_count_, -1))

            weights_are_all_positive_ = min(svm.atts.coefficients) >= 0

            # SVM part
            if vector_count_ == 0 and mode_ == "SVM_LINEAR":
                res = np.empty((X.shape[0], class_count_), dtype=X.dtype)
                for n in range(X.shape[0]):
                    scores = self._run_linear(X[n], coefs, class_count_, kernel_type_)
                    res[n, :] = scores
            else:
                res = np.empty(
                    (X.shape[0], class_count_ * (class_count_ - 1) // 2), dtype=X.dtype
                )
                votes = np.empty((X.shape[0], class_count_), dtype=X.dtype)
                for n in range(X.shape[0]):
                    vote, scores = self._run_svm(
                        X[n],
                        sv,
                        vector_count_,
                        kernel_type_,
                        class_count_,
                        starting_vector_,
                        coefs,
                    )
                    res[n, :] = scores
                    votes[n, :] = vote

            # proba
            if (
                self._svm.atts.prob_a is not None
                and len(self._svm.atts.prob_a) > 0
                and mode_ == "SVM_SVC"
            ):
                scores = np.empty((res.shape[0], class_count_), dtype=X.dtype)
                for n in range(scores.shape[0]):
                    s = self._probabilities(res[n], class_count_)
                    scores[n, :] = s
                has_proba = True
            else:
                scores = res
                has_proba = False

            # finalization
            final_scores = None
            labels = []
            for n in range(scores.shape[0]):
                label, new_scores = self._compute_final_scores(
                    votes[n],
                    scores[n],
                    weights_are_all_positive_,
                    has_proba,
                    classlabels_ints,
                )
                if final_scores is None:
                    final_scores = np.empty(
                        (X.shape[0], new_scores.size), dtype=X.dtype
                    )
                final_scores[n, :] = new_scores
                labels.append(label)

            # labels
            if classlabels_strings is not None and len(classlabels_strings) > 0:
                return (
                    np.array([classlabels_strings[i] for i in labels]),
                    final_scores,
                )
            return (np.array(labels, dtype=np.int64), final_scores)
