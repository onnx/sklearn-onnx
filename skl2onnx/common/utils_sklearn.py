# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from collections import OrderedDict
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def enumerate_model_names(model, prefix="", short=True):
    """
    Enumerates ``tuple (name, model)`` associated
    to the model itself.
    """
    if isinstance(model, (list, tuple)):
        if all(map(lambda x: isinstance(x, tuple) and len(x) in (2, 3),
                   model)):
            for i, named_mod in enumerate(model):
                name, mod = named_mod[:2]
                p = (name if short and prefix == ""
                     else "{}__{}".format(prefix, name))
                for t in enumerate_model_names(mod, p, short=short):
                    yield t
        else:
            for i, mod in enumerate(model):
                p = (i if short and prefix == ""
                     else "{}__{}".format(prefix, i))
                for t in enumerate_model_names(mod, p, short=short):
                    yield t
    elif isinstance(model, (dict, OrderedDict)):
        for name, mod in model.items():
            p = (name if short and prefix == ""
                 else "{}__{}".format(prefix, name))
            for t in enumerate_model_names(mod, p, short=short):
                yield t
    else:
        yield (prefix, model)
        reserved_atts = {'transformers', 'steps', 'transformer_list',
                         'named_estimators_', 'named_transformers_',
                         'transformer_', 'estimator_'}
        for key in dir(model):
            if (key in ('estimators_', 'estimator') and
                    hasattr(model, 'named_estimators_')):
                continue
            if (key in ('transformers_', 'transformers') and
                    hasattr(model, 'named_transformers_')):
                continue
            if (key in reserved_atts or
                    (key.endswith("_") and not key.endswith("__") and
                     not key.startswith('_'))):
                try:
                    obj = getattr(model, key)
                except AttributeError:
                    continue
                if (hasattr(obj, 'get_params') and
                        isinstance(obj, BaseEstimator)):
                    prefix = (key if short and prefix == ""
                              else "{}__{}".format(prefix, key))
                    yield (prefix, obj)
                elif isinstance(obj, (list, tuple, dict, OrderedDict)):
                    if not short or key not in reserved_atts:
                        prefix = (key if short and prefix == ""
                                  else "{}__{}".format(prefix, key))
                    for t in enumerate_model_names(obj, prefix, short=short):
                        yield t


def has_pipeline(model):
    """
    Tells if a model contains a pipeline.
    """
    return any(map(lambda x: isinstance(x[1], Pipeline),
                   enumerate_model_names(model)))
