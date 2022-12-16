# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.special import expit
from onnx.defs import onnx_opset_version


if onnx_opset_version() >= 18:
    from onnx.reference.op_run import OpRun

    class Scaler(OpRun):

        op_domain = "ai.onnx.ml"

        def _run(self, x, offset=None, scale=None):
            dx = x - offset
            return (dx * scale,)

    class LinearClassifier(OpRun):

        op_domain = "ai.onnx.ml"

        @staticmethod
        def _post_process_predicted_label(
            label, scores, classlabels_ints_string
        ):
            """
            Replaces int64 predicted labels by the corresponding
            strings.
            """
            if classlabels_ints_string is not None:
                label = np.array([classlabels_ints_string[i] for i in label])
            return label, scores

        def _run(
            self,
            x,
            classlabels_ints=None,
            classlabels_strings=None,
            coefficients=None,
            intercepts=None,
            multi_class=None,
            post_transform=None,
        ):
            coefficients = np.array(coefficients).astype(x.dtype)
            intercepts = np.array(intercepts).astype(x.dtype)
            n_class = max(
                len(classlabels_ints or []), len(classlabels_strings or [])
            )
            n = coefficients.shape[0] // n_class
            coefficients = coefficients.reshape(n_class, n).T
            scores = np.dot(x, coefficients)
            if intercepts is not None:
                scores += intercepts

            if post_transform == "NONE":
                pass
            elif post_transform == "LOGISTIC":
                scores = expit(scores)
            elif post_transform == "SOFTMAX":
                np.subtract(
                    scores, scores.max(axis=1)[:, np.newaxis], out=scores
                )
                scores = np.exp(scores)
                scores = np.divide(
                    scores, scores.sum(axis=1)[:, np.newaxis]
                )
            else:
                raise NotImplementedError(  # pragma: no cover
                    f"Unknown post_transform: '{post_transform}'."
                )

            if coefficients.shape[1] == 1:
                labels = np.zeros((scores.shape[0],), dtype=x.dtype)
                labels[scores > 0] = 1
            else:
                labels = np.argmax(scores, axis=1)
            if classlabels_ints is not None:
                labels = np.array(
                    [classlabels_ints[i] for i in labels], dtype=np.int64
                )
            elif classlabels_strings is not None:
                labels = np.array([classlabels_strings[i] for i in labels])
            return (labels, scores)

    class LinearRegressor(OpRun):

        op_domain = "ai.onnx.ml"

        def _run(
            self,
            x,
            coefficients=None,
            intercepts=None,
            targets=1,
            post_transform=None,
        ):
            coefficients = np.array(coefficients).astype(x.dtype)
            intercepts = np.array(intercepts).astype(x.dtype)
            n = coefficients.shape[0] // targets
            coefficients = coefficients.reshape(targets, n).T
            score = np.dot(x, coefficients)
            if self.intercepts is not None:
                score += intercepts
            if post_transform == "NONE":
                pass
            else:
                raise NotImplementedError(  # pragma: no cover
                    f"Unknown post_transform: '{self.post_transform}'."
                )
            return (score,)

    class Normalizer(OpRun):

        op_domain = "ai.onnx.ml"

        @staticmethod
        def norm_max(x):
            "max normalization"
            div = np.abs(x).max(axis=1).reshape((x.shape[0], -1))
            return x / np.maximum(div, 1e-30)

        @staticmethod
        def norm_l1(x):
            "L1 normalization"
            div = np.abs(x).sum(axis=1).reshape((x.shape[0], -1))
            return x / np.maximum(div, 1e-30)

        @staticmethod
        def norm_l2(x):
            "L2 normalization"
            xn = np.square(x).sum(axis=1)
            np.sqrt(xn, out=xn)
            norm = np.maximum(xn.reshape((x.shape[0], -1)), 1e-30)
            return x / norm

        def _run(self, x, norm=None):
            if norm == "MAX":
                _norm = Normalizer.norm_max
            elif self.norm == "L1":
                _norm = Normalizer.norm_l1
            elif self.norm == "L2":
                _norm = Normalizer.norm_l2
            else:
                raise ValueError(  # pragma: no cover
                    f"Unexpected value for norm='{norm}'."
                )
            return (_norm(x),)

    class OneHotEncoder(OpRun):

        op_domain = "ai.onnx.ml"

        def _run(self, x, cats_int64s=None, cats_strings=None, zeros=None):
            if len(cats_int64s) > 0:
                classes_ = {v: i for i, v in enumerate(cats_int64s)}
            elif len(cats_strings) > 0:
                classes_ = {
                    v.decode("utf-8"): i for i, v in enumerate(cats_strings)
                }
            else:
                raise RuntimeError("No encoding was defined.")

            shape = x.shape
            new_shape = shape + (len(classes_),)
            res = np.zeros(new_shape, dtype=np.float32)
            if len(x.shape) == 1:
                for i, v in enumerate(x):
                    j = classes_.get(v, -1)
                    if j >= 0:
                        res[i, j] = 1.0
            elif len(x.shape) == 2:
                for a, row in enumerate(x):
                    for i, v in enumerate(row):
                        j = classes_.get(v, -1)
                        if j >= 0:
                            res[a, i, j] = 1.0
            else:
                raise RuntimeError(
                    f"This operator is not implemented "
                    f"for " f"shape {x.shape}.")

            if not self.zeros:
                red = res.sum(axis=len(res.shape) - 1)
                if np.min(red) == 0:
                    rows = []
                    for i, val in enumerate(red):
                        if val == 0:
                            rows.append(dict(row=i, value=x[i]))
                            if len(rows) > 5:
                                break
                    raise RuntimeError(  # pragma no cover
                        "One observation did not have any "
                        "defined category.\n"
                        "classes: {}\nfirst rows:\n{}\nres:\n{}\nx:"
                        "\n{}".format(
                            self.classes_,
                            "\n".join(str(_) for _ in rows),
                            res[:5],
                            x[:5],
                        )
                    )

            return (res,)

    class Binarizer(OpRun):

        op_domain = "ai.onnx.ml"

        def _run(self, x, threshold=None):
            X = x.copy()
            cond = X > self.threshold
            not_cond = np.logical_not(cond)
            X[cond] = 1
            X[not_cond] = 0
            return (X,)
