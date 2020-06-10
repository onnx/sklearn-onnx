# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import _deprecate_positional_args


class CastTransformer(TransformerMixin, BaseEstimator):

    """
    Cast features into a specific types.
    This should be used to minimize the conversion
    of a pipeline using float32 instead of double.
    
    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    """  # noqa

    @_deprecate_positional_args
    def __init__(self, *, dtype=np.float32):
        self.dtype = dtype

    def _cast(self, a, name):
        if not isinstance(a, np.ndarray):
            raise TypeError("{} must be a numpy array.".format(name))
        try:
            a2 = a.astype(self.dtype)
        except ValueError:
            raise ValueError(
                "Unable to cast {} from {} into {}.".format(
                    name, a.dtype, self.dtype))
        return a2

    def fit(self, X, y=None, sample_weight=None):
        """
        Does nothing except checking *dtype* may be applied.
        """
        self._cast(X, 'X')
        return self

    def transform(self, X, y=None):
        """
        Casts array X.
        """
        return self._cast(X, 'X')
