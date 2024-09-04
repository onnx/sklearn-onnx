# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.special import expit
from scipy.sparse import coo_matrix
from onnx.defs import onnx_opset_version


if 19 >= onnx_opset_version() >= 18:
    from onnx.reference.op_run import OpRun

    class FusedMatMul(OpRun):
        @staticmethod
        def _fmatmul00(a, b, alpha):
            return np.matmul(a, b) * alpha

        @staticmethod
        def _fmatmul01(a, b, alpha):
            return np.matmul(a, b.T) * alpha

        @staticmethod
        def _fmatmul10(a, b, alpha):
            return np.matmul(a.T, b) * alpha

        @staticmethod
        def _fmatmul11(a, b, alpha):
            return np.matmul(a.T, b.T) * alpha

        @staticmethod
        def _transpose(x, trans, transBatch):
            if trans:
                n = len(x.shape)
                perm = list(range(n - 2)) + [n - 2, n - 1]
                x = np.transpose(x, perm)
            if transBatch:
                n = len(x.shape)
                perm = list(range(1, n - 2)) + [0, n - 1]
                x = np.transpose(x, perm)
            return x

        def _run(
            self,
            a,
            b,
            alpha=None,
            transA=None,
            transB=None,
            transBatchA=None,
            transBatchB=None,
        ):
            if transA:
                _meth = FusedMatMul._fmatmul11 if transB else FusedMatMul._fmatmul10
            else:
                _meth = FusedMatMul._fmatmul01 if transB else FusedMatMul._fmatmul00
            _meth = lambda a, b: _meth(a, b, alpha)
            # more recent versions of the operator
            if transBatchA is None:
                transBatchA = 0
            if transBatchB is None:
                transBatchB = 0

            if transBatchA or transBatchB or len(a.shape) != 2 or len(b.shape) != 2:
                ta = self._transpose(a, transA, transBatchA)
                tb = self._transpose(b, transB, transBatchB)
                try:
                    return (np.matmul(ta, tb) * alpha,)
                except ValueError as e:
                    raise ValueError(
                        f"Unable to multiply shape {a.shape}x{b.shape} "
                        f"({ta.shape}x{tb.shape}) "
                        f"with transA={transA}, "
                        f"transB={transB}, "
                        f"transBatchA={transBatchA}, "
                        f"transBatchB={transBatchB}, "
                        f"meth={_meth}."
                    ) from e
            try:
                return (_meth(a, b),)
            except ValueError as e:
                raise ValueError(
                    f"Unable to multiply shape {a.shape}x{b.shape} "
                    f"with transA={transA}, "
                    f"transB={transB}, "
                    f"transBatchA={transBatchA}, "
                    f"transBatchB={transBatchB}, "
                    f"meth={_meth}."
                ) from e

    class Scaler(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(self, x, offset=None, scale=None):
            dx = x - offset
            return (dx * scale,)

    class LinearClassifier(OpRun):
        op_domain = "ai.onnx.ml"

        @staticmethod
        def _post_process_predicted_label(label, scores, classlabels_ints_string):
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
            dtype = x.dtype
            if dtype != np.float64:
                x = x.astype(np.float32)
            coefficients = np.array(coefficients).astype(x.dtype)
            intercepts = np.array(intercepts).astype(x.dtype)
            n_class = max(len(classlabels_ints or []), len(classlabels_strings or []))
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
                np.subtract(scores, scores.max(axis=1)[:, np.newaxis], out=scores)
                scores = np.exp(scores)
                scores = np.divide(scores, scores.sum(axis=1)[:, np.newaxis])
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
                labels = np.array([classlabels_ints[i] for i in labels], dtype=np.int64)
            elif classlabels_strings is not None:
                labels = np.array([classlabels_strings[i] for i in labels])
            return (labels, scores)

    class LinearRegressor(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(
            self, x, coefficients=None, intercepts=None, targets=1, post_transform=None
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
                raise NotImplementedError(
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
            if cats_int64s is not None and len(cats_int64s) > 0:
                classes_ = {v: i for i, v in enumerate(cats_int64s)}
            elif len(cats_strings) > 0:
                classes_ = {v: i for i, v in enumerate(cats_strings)}
            else:
                raise RuntimeError("No encoding was defined.")

            shape = x.shape
            new_shape = (*shape, len(classes_))
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
                    f"This operator is not implemented for shape {x.shape}."
                )

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

    class FeatureVectorizer(OpRun):
        op_domain = "ai.onnx.ml"

        def _preprocess(self, a, axis):
            if axis >= len(a.shape):
                new_shape = a.shape + (1,) * (axis + 1 - len(a.shape))
                return a.reshape(new_shape)
            return a

        def _run(self, *args, inputdimensions=None):
            args = [self._preprocess(a, axis) for a, axis in zip(args, inputdimensions)]
            dimensions = set(inputdimensions)
            if len(set(dimensions)) == 1:
                res = np.concatenate(args, axis=inputdimensions[0])
                return (res,)
            raise RuntimeError(
                f"inputdimensions={inputdimensions} is not supported yet."
            )

    class Imputer(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(
            self,
            x,
            imputed_value_floats=None,
            imputed_value_int64s=None,
            replaced_value_float=None,
            replaced_value_int64=None,
        ):
            if imputed_value_floats is not None and len(imputed_value_floats) > 0:
                values = imputed_value_floats
                replace = replaced_value_float
            elif imputed_value_int64s is not None and len(imputed_value_int64s) > 0:
                values = imputed_value_int64s
                replace = replaced_value_int64
            else:
                raise ValueError("Missing are not defined.")

            if isinstance(values, list):
                values = np.array(values)
            if len(x.shape) != 2:
                raise TypeError(f"x must be a matrix but shape is {x.shape}")
            if values.shape[0] not in (x.shape[1], 1):
                raise TypeError(  # pragma: no cover
                    f"Dimension mismatch {values.shape[0]} != {x.shape[1]}"
                )
            x = x.copy()
            if np.isnan(replace):
                for i in range(0, x.shape[1]):
                    val = values[min(i, values.shape[0] - 1)]
                    x[np.isnan(x[:, i]), i] = val
            else:
                for i in range(0, x.shape[1]):
                    val = values[min(i, values.shape[0] - 1)]
                    x[x[:, i] == replace, i] = val

            return (x,)

    class LabelEncoder(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(
            self,
            x,
            default_float=None,
            default_int64=None,
            default_string=None,
            keys_floats=None,
            keys_int64s=None,
            keys_strings=None,
            values_floats=None,
            values_int64s=None,
            values_strings=None,
        ):
            keys = keys_floats or keys_int64s or keys_strings
            values = values_floats or values_int64s or values_strings
            classes = {k: v for k, v in zip(keys, values)}
            if id(keys) == id(keys_floats):
                cast = float
            elif id(keys) == id(keys_int64s):
                cast = int
            else:
                cast = str
            if id(values) == id(values_floats):
                defval = default_float
                dtype = np.float32
            elif id(values) == id(values_int64s):
                defval = default_int64
                dtype = np.int64
            else:
                defval = default_string
                if not isinstance(defval, str):
                    defval = ""
                dtype = np.str_
            shape = x.shape
            if len(x.shape) > 1:
                x = x.flatten()
            res = []
            for i in range(0, x.shape[0]):
                v = classes.get(cast(x[i]), defval)
                res.append(v)
            return (np.array(res, dtype=dtype).reshape(shape),)

    class DictVectorizer(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(self, x, int64_vocabulary=None, string_vocabulary=None):
            if isinstance(x, (np.ndarray, list)):
                dict_labels = {}
                if int64_vocabulary:
                    for i, v in enumerate(int64_vocabulary):
                        dict_labels[v] = i
                else:
                    for i, v in enumerate(string_vocabulary):
                        dict_labels[v] = i
                if len(dict_labels) == 0:
                    raise RuntimeError(
                        "int64_vocabulary and string_vocabulary "
                        "cannot be both empty."
                    )

                values = []
                rows = []
                cols = []
                for i, row in enumerate(x):
                    for k, v in row.items():
                        values.append(v)
                        rows.append(i)
                        cols.append(dict_labels[k])
                values = np.array(values)
                rows = np.array(rows)
                cols = np.array(cols)
                return (
                    coo_matrix(
                        (values, (rows, cols)), shape=(len(x), len(dict_labels))
                    ).todense(),
                )

            if isinstance(x, dict):
                keys = int64_vocabulary or string_vocabulary
                res = []
                for k in keys:
                    res.append(x.get(k, 0))
                return (np.array(res),)

            raise TypeError(f"x must be iterable not {type(x)}.")  # pragma: no cover
