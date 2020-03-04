# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from inspect import signature


def check_signature(fct, reference, skip=None):
    """
    Checks that two functions have the same signature
    (same parameter names).
    Raises an exception otherwise.
    """
    sig = signature(fct)
    sig_ref = signature(reference)
    if len(sig.parameters) != len(sig_ref.parameters):
        raise TypeError(
            "Function '{}' must have {} parameters but has {}."
            "".format(fct.__name__, len(sig_ref.parameters),
                      len(sig.parameters)))
    for i, (a, b) in enumerate(zip(sig.parameters, sig_ref.parameters)):
        if a != b and skip is not None and b not in skip and a not in skip:
            raise NameError(
                "Parameter name mismatch at position {}."
                "Function '{}' has '{}' but '{}' is expected."
                "".format(i + 1, fct.__name__, a, b))
