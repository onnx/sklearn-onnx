# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np

from .tests_helper import dump_data_and_model  # noqa
from .tests_helper import (  # noqa
    dump_one_class_classification,
    dump_binary_classification,
    dump_multiple_classification,
)
from .tests_helper import (  # noqa
    dump_multiple_regression,
    dump_single_regression,
)


def create_tensor(N, C, H=None, W=None):
    if H is None and W is None:
        return np.random.rand(N, C).astype(np.float32, copy=False)
    elif H is not None and W is not None:
        return np.random.rand(N, C, H, W).astype(np.float32, copy=False)
    else:
        raise ValueError('This function only produce 2-D or 4-D tensor')
