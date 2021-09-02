# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
try:
    from sklearn.utils.validation import _deprecate_positional_args
except ImportError:
    def _deprecate_positional_args(x): return x  # noqa


class WOETransformer(TransformerMixin, BaseEstimator):

    """
    This transformer cannot be trained. It takes a list of intervals,
    one list per columns, and returns for every feature the list
    of intervals this features falls into. This transformer is close
    to `WOEEncoder <https://contrib.scikit-learn.org/category_encoders/
    woe.html?highlight=woe#category_encoders.woe.WOEEncoder>`_.
    *sklearn-onnx* only converts *scikit-learn* models but this class
    can be used as an example for other converters converting similar
    transforms.

    Parameters
    ----------
    intervals : list of list of tuples,
        every tuple is an interval, one list per column,
        a list can be replaced with constant `'passthrough'`
        which leaves the features untouched.

    An interval is defined with four values `(a, b, False, True)`:
    * `(a, b, False, False)` means `]a, b[`
    * `(a, b, False, True)` means `]a, b]`
    * `(a, b, True, False)` means `[a, b[`
    * `(a, b, True, True)` means `[a, b]`
    Boolean defines if the extremity belongs to the interval or not.
    By default `(a, b)` is equivalent to `(a, b, False, True)`.
    """

    @_deprecate_positional_args
    def __init__(self, intervals=None):
        self.intervals = intervals

    def fit(self, X, y=None, sample_weight=None):
        """
        Does nothing except checking *dtype* may be applied.
        """
        self.intervals_ = []
        dim = 0
        self.indices_ = []
        for i in range(X.shape[1]):
            if i >= len(self.intervals):
                self.intervals_.append(None)
                self.indices_.append((dim, dim + 1))
                dim += 1
                continue
            intervals = self.intervals[i]
            if intervals == 'passthrough':
                self.intervals_.append(None)
                self.indices_.append((dim, dim + 1))
                dim += 1
                continue
            if not isinstance(intervals, list):
                raise TypeError(
                    "Intervals for column %d must be a list not %r."
                    "" % (i, intervals))
            inlist = []
            for index, interval in enumerate(intervals):
                if not isinstance(interval, tuple):
                    raise TypeError(
                        "Interval %d is not a tuple but %r." % (i, interval))
                if len(interval) < 2:
                    raise ValueError(
                        "Interval %d should have at least two values "
                        "%r." % interval)
                res = []
                for j in range(0, 2):
                    if not isinstance(interval[j], float):
                        raise TypeError(
                            "Value at index %i in %r must be a float."
                            "" % (
                                j, interval))
                    res.append(interval[j])
                if len(interval) >= 3:
                    if not isinstance(interval[2], bool):
                        raise TypeError(
                            "Value at index %i in %r must be a boolean."
                            "" % (2, interval))
                    res.append(interval[2])
                else:
                    res.append(False)
                if len(interval) >= 4:
                    if not isinstance(interval[3], bool):
                        raise TypeError(
                            "Value at index %i in %r must be a boolean."
                            "" % (3, interval))
                    res.append(interval[3])
                else:
                    res.append(True)
                inlist.append(tuple(res))

            self.intervals_.append(inlist)
            self.indices_.append((dim, dim + len(inlist)))
            dim += len(inlist)

        self.n_dims_ = dim
        return self

    def _transform_column(self, X, i):
        col = X[:, i]
        intervals = self.intervals_[i]
        if intervals is None:
            return col.reshape((-1, 1))
        res = np.zeros((X.shape[0], len(intervals)), dtype=X.dtype)
        for i, interval in enumerate(intervals):
            if interval[2]:
                left = col >= interval[0]
            else:
                left = col > interval[0]
            if interval[3]:
                right = col <= interval[1]
            else:
                right = col < interval[1]
            res[:, i] = (left * right).astype(X.dtype)
        return res

    def transform(self, X, y=None):
        """
        Applies the transformation.
        """
        res = np.zeros((X.shape[0], self.n_dims_), dtype=X.dtype)
        for i in range(X.shape[1]):
            a, b = self.indices_[i]
            res[:, a: b] = self._transform_column(X, i)
        return res

    def get_feature_names(self):
        """
        Returns the features names.
        """
        names = []
        for i, intervals in enumerate(self.intervals_):
            if intervals is None:
                names.append("X%d" % i)
                continue
            for interval in intervals:
                name = [
                    "[" if interval[2] else "]",
                    str(interval[0]), ",", str(interval[1]),
                    "]" if interval[3] else "["]
                names.append("".join(name))
        return names

    def _decision_thresholds(self, add_index=False):
        "Returns all decision thresholds."
        extremities = []
        for intervals in self.intervals_:
            if intervals is None:
                extremities.append(None)
                continue
            thresholds = []
            for index, interval in enumerate(intervals):
                if add_index:
                    thresholds.append((interval[0], not interval[2], index))
                    thresholds.append((interval[1], interval[3], index))
                else:
                    thresholds.append((interval[0], not interval[2]))
                    thresholds.append((interval[1], interval[3]))
            extremities.append(list(sorted(set(thresholds))))
        return extremities
