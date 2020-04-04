# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Main entry point to the converter from the *scikit-learn* to *onnx*.
"""
__version__ = "1.6.1"
__author__ = "Microsoft"
__producer__ = "skl2onnx"
__producer_version__ = __version__
__domain__ = "ai.onnx"
__model_version__ = 0


from .convert import convert_sklearn, to_onnx, wrap_as_onnx_mixin # noqa
from ._supported_operators import update_registered_converter # noqa
from ._parse import update_registered_parser # noqa


def supported_converters(from_sklearn=False):
    """
    Returns the list of supported converters.
    To find the converter associated to a specific model,
    the library gets the name of the model class,
    adds ``'Sklearn'`` as a prefix and retrieves
    the associated converter if available.

    :param from_sklearn: every supported model is mapped to converter
        by a name prefixed with ``'Sklearn'``, the prefix is removed
        if this parameter is False but the function only returns converters
        whose name is prefixed by ``'Sklearn'``
    :return: list of supported models as string
    """
    from .common._registration import _converter_pool # noqa
    # The two following lines populates the list of supported converters.
    from . import shape_calculators # noqa
    from . import operator_converters # noqa

    names = sorted(_converter_pool.keys())
    if from_sklearn:
        return [_[7:] for _ in names if _.startswith('Sklearn')]
    else:
        return list(names)
