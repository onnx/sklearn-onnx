import numpy as np
from sklearn.base import BaseEstimator


class DictWrapper(BaseEstimator):
    """
    A wrapper for dictionary data that provides transformation and fitting capabilities.

    Parameters
    ----------
    data : dict
        The dictionary data to be wrapped.

    Attributes
    ----------
    data : dict
        The dictionary data to be wrapped.
    """

    def __init__(self, data):
        self.data = data

    def transform(self, X):
        """
        Transforms the input by returning the corresponding value from the dictionary.

        Parameters
        ----------
        X : str
            The key to look up in the dictionary.

        Returns
        -------
        Any
            The value corresponding to the key in the dictionary.
        """
        return self.data.get(X)

    def fit(self, X, y=None):
        """
        A placeholder fit method. Does nothing.

        Parameters
        ----------
        X : array-like
            The input samples.
        y : array-like, default=None
            The target values. Ignored.
        """
        pass

    def __len__(self):
        """
        Returns the length of the first value in the dictionary.

        Returns
        -------
        int
            The length of the first value in the dictionary.
        """
        return len(list(self.data.values())[0])


def fill_missing_indices(d, max_key, scale=10, posterior_len=2, min_key=0):
    """
    Fills missing indices in a dictionary with a default value.

    Parameters
    ----------
    d : dict
        The dictionary to fill.
    max_key : int
        The maximum key value to consider.
    scale : int, default=10
        The scale factor for the range of keys.
    posterior_len : int, default=2
        The length of the posterior array to use as the default value.
    min_key : int, default=0
        The minimum key value to consider.

    Returns
    -------
    dict
        The filled dictionary.

    Raises
    ------
    ZeroDivisionError
        If `posterior_len` is zero.

    Notes
    -----
    This function creates a copy of the input dictionary and fills in missing keys in the range from `min_key`
    to `max_key * scale` with a default value. The default value is an array of length `posterior_len` with all
    elements equal to `1 / posterior_len`. If `posterior_len` is zero, a `ZeroDivisionError` is raised and logged.
    """
    try:
        posteriors = np.ones(posterior_len, dtype=np.float32) / np.float32(
            posterior_len
        )
    except ZeroDivisionError as zde:
        raise zde
    d_cp = d.copy()  # copy to make sure that the original keys are not altered
    min_key, max_key = min_key, max_key
    for i in range(min_key, max_key * scale):
        if i not in d_cp:
            d_cp[i] = posteriors
    return d_cp
