# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from .proto import get_opset_number_from_onnx
from .common._topology import convert_topology
from ._parse import parse_sklearn_model

# Invoke the registration of all our converters and shape calculators.
from . import shape_calculators # noqa
from . import operator_converters # noqa


def convert_sklearn(model, name=None, initial_types=None, doc_string='',
                    target_opset=None, custom_conversion_functions=None,
                    custom_shape_calculators=None,
                    custom_parsers=None, options=None,
                    intermediate=False):
    """
    This function produces an equivalent ONNX model of the given scikit-learn model.
    The supported converters is returned by function
    :func:`supported_converters <skl2onnx.supported_converters>`.

    For pipeline conversion, user needs to make sure each component
    is one of our supported items.
    This function converts the specified *scikit-learn* model into its *ONNX* counterpart.
    Note that for all conversions, initial types are required.
    *ONNX* model name can also be specified.

    :param model: A scikit-learn model
    :param initial_types: a python list. Each element is a tuple of a variable name
        and a type defined in data_types.py
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function,
        it takes precedence over registered converters
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
        it takes precedence over registered shape calculators.
    :param custom_parsers: parsers determines which outputs is expected for which particular task,
        default parsers are defined for classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary ``{ type: fct_parser(scope, model, inputs, custom_parsers=None) }``
    :param options: specific options given to converters (see :ref:`l-conv-options`)
    :param intermediate: if True, the function returns the converted model and , and :class:`Topology`,
        it returns the converted model otherwise
    :return: An ONNX model (type: ModelProto) which is equivalent to the input scikit-learn model

    Example of initial_types:
    Assume that the specified *scikit-learn* model takes a heterogeneous list as its input.
    If the first 5 elements are floats and the last 10 elements are integers,
    we need to specify initial types as below. The [1] in [1, 5] indicates
    the batch size here is 1.

    ::

        from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
        initial_type = [('float_input', FloatTensorType([1, 5])),
                        ('int64_input', Int64TensorType([1, 10]))]

    .. note::

        If a pipeline includes an instance of
        `ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_,
        *scikit-learn* allow the user to specify columns by names. This option is not supported
        by *sklearn-onnx* as features names could be different in input data and the ONNX graph
        (defined by parameter *initial_types*), only integers are supported.

    .. _l-conv-options:

    Converters options
    ++++++++++++++++++

    Some ONNX operators exposes parameters *sklearn-onnx* cannot
    guess from the raw model. Some default values are usually suggested
    but the users may have to manually overwrite them. This need
    is not obvious to do when a model is included in a pipeline.
    That's why these options can be given to function *convert_sklearn*
    as a dictionary ``{model_type: parameters in a dictionary}`` or
    ``{model_id: parameters in a dictionary}``.
    Option *sep* is used to specify the delimiters between two words
    when the ONNX graph needs to tokenize a string.
    The default value is short and may not include all
    the necessary values. It can be overwritten as:

    ::

        extra = {TfidfVectorizer: {"sep": [' ', '.', '?', ',', ';', ':', '!', '(', ')']}}
        model_onnx = convert_sklearn(model, "tfidf",
                                     initial_types=[("input", StringTensorType([1, 1]))],
                                     options=extra)

    But if a pipeline contains two model of the same class,
    it is possible to disintguish between the two with function *id*:

    ::

        extra = {id(model): {"sep": [' ', '.', '?', ',', ';', ':', '!', '(', ')']}}
        model_onnx = convert_sklearn(pipeline, "pipeline-with-2-tfidf",
                                     initial_types=[("input", StringTensorType([1, 1]))],
                                     options=extra)

    It is used in example :ref:`l-example-tfidfvectorizer`.
    """ # noqa
    if initial_types is None:
        if hasattr(model, 'infer_initial_types'):
            initial_types = model.infer_initial_types()
        else:
            raise ValueError('Initial types are required. See usage of '
                             'convert(...) in skl2onnx.convert for details')

    if name is None:
        name = str(uuid4().hex)

    target_opset = (target_opset
                    if target_opset else get_opset_number_from_onnx())
    # Parse scikit-learn model as our internal data structure
    # (i.e., Topology)
    topology = parse_sklearn_model(model, initial_types, target_opset,
                                   custom_conversion_functions,
                                   custom_shape_calculators, custom_parsers)

    # Infer variable shapes
    topology.compile()

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name, doc_string, target_opset,
                                  options=options)

    return (onnx_model, topology) if intermediate else onnx_model


def to_onnx(model, X=None, name=None, initial_types=None):
    """
    Calls :func:`convert_sklearn` with simplified parameters.

    :param model: model to convert
    :param X: training set, can be None, it is used to infered the
        input types (*initial_types*)
    :param initial_types: if X is None, then *initial_types* must be
        defined
    :param name: name of the model
    :return: converted model

    This function checks if the model inherits from class
    :class:`OnnxOperatorMixin`, it calls method *to_onnx*
    in that case otherwise it calls :func:`convert_sklearn`.
    """
    from .algebra.onnx_operator_mixin import OnnxOperatorMixin
    from .algebra.type_helper import guess_initial_types

    if isinstance(model, OnnxOperatorMixin):
        return model.to_onnx(X=X, name=name)
    if name is None:
        name = model.__class__.__name__
    initial_types = guess_initial_types(X, initial_types)
    return convert_sklearn(model, initial_types=initial_types, name=name)


def wrap_as_onnx_mixin(model):
    """
    Combines a *scikit-learn* class with :class:`OnnxOperatorMixin`
    which produces a new object which combines *scikit-learn* API
    and *OnnxOperatorMixin* API.
    """
    from .algebra.sklearn_ops import find_class
    cl = find_class(model.__class__)
    if "automation" in str(cl):
        raise RuntimeError("Wrong class name '{}'.".format(cl))
    state = model.__getstate__()
    obj = object.__new__(cl)
    obj.__setstate__(state)
    return obj
