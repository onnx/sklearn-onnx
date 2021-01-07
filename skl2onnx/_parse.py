# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np

from sklearn import pipeline
from sklearn.base import (
    ClassifierMixin, ClusterMixin, is_classifier
)
try:
    from sklearn.base import OutlierMixin
except ImportError:
    # scikit-learn <= 0.19
    class OutlierMixin:
        pass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC, SVC
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # ColumnTransformer was introduced in 0.20.
    ColumnTransformer = None

from ._supported_operators import (
    _get_sklearn_operator_name, cluster_list, outlier_list)
from ._supported_operators import (
    sklearn_classifier_list, sklearn_operator_name_map)
from .common._container import SklearnModelContainerNode
from .common._registration import _converter_pool, _shape_calculator_pool
from .common._topology import Topology
from .common.data_types import (
    DictionaryType, Int64TensorType, SequenceType,
    StringTensorType, TensorType, guess_tensor_type)
from .common.utils import get_column_indices
from .common.utils_checking import check_signature
from .common.utils_classifier import get_label_classes
from .common.utils_sklearn import has_pipeline, _process_options


do_not_merge_columns = tuple(
    filter(lambda op: op is not None,
           [OneHotEncoder, ColumnTransformer]))


def _fetch_input_slice(scope, inputs, column_indices):
    if not isinstance(inputs, list):
        raise TypeError("Parameter inputs must be a list.")
    if len(inputs) == 0:
        raise RuntimeError("Operator ArrayFeatureExtractor requires at "
                           "least one inputs.")
    if len(inputs) != 1:
        raise RuntimeError("Operator ArrayFeatureExtractor does not support "
                           "multiple input tensors.")
    if (isinstance(inputs[0].type, TensorType) and
            len(inputs[0].type.shape) == 2 and
            inputs[0].type.shape[1] == len(column_indices)):
        # No need to extract.
        return inputs
    array_feature_extractor_operator = scope.declare_local_operator(
                                            'SklearnArrayFeatureExtractor')
    array_feature_extractor_operator.inputs = inputs
    array_feature_extractor_operator.column_indices = column_indices
    output_variable_name = scope.declare_local_variable(
                            'extracted_feature_columns', inputs[0].type)
    array_feature_extractor_operator.outputs.append(output_variable_name)
    return array_feature_extractor_operator.outputs


def _parse_sklearn_simple_model(scope, model, inputs, custom_parsers=None):
    """
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., *OneHotEncoder*
        or *LogisticRegression*)
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next
        stage
    """
    # alias can be None
    if isinstance(model, str):
        raise RuntimeError("Parameter model must be an object not a "
                           "string '{0}'.".format(model))
    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs

    if hasattr(model, 'onnx_parser'):
        parser_names = model.onnx_parser(scope=scope, inputs=inputs)
        if parser_names is not None:
            names = parser_names()
            for name in names:
                var = scope.declare_local_variable(
                    name, guess_tensor_type(inputs[0].type))
                this_operator.outputs.append(var)
            return this_operator.outputs

    if (type(model) in sklearn_classifier_list
            or isinstance(model, ClassifierMixin)
            or (isinstance(model, GridSearchCV)
                and is_classifier(model))):
        # For classifiers, we may have two outputs, one for label and
        # the other one for probabilities of all classes. Notice that
        # their types here are not necessarily correct and they will
        # be fixed in shape inference phase.
        label_variable = scope.declare_local_variable(
            'label', Int64TensorType())
        probability_tensor_variable = scope.declare_local_variable(
            'probabilities', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_tensor_variable)

    elif type(model) in cluster_list or isinstance(model, ClusterMixin):
        # For clustering, we may have two outputs, one for label and
        # the other one for scores of all classes. Notice that their
        # types here are not necessarily correct and they will be fixed
        # in shape inference phase
        label_variable = scope.declare_local_variable(
            'label', Int64TensorType())
        score_tensor_variable = scope.declare_local_variable(
            'scores', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(score_tensor_variable)

    elif type(model) in outlier_list or isinstance(model, OutlierMixin):
        # For clustering, we may have two outputs, one for label and
        # the other one for scores.
        label_variable = scope.declare_local_variable(
            'label', Int64TensorType())
        score_tensor_variable = scope.declare_local_variable(
            'scores', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(score_tensor_variable)

    elif type(model) == NearestNeighbors:
        # For Nearest Neighbours, we have two outputs, one for nearest
        # neighbours' indices and the other one for distances
        index_variable = scope.declare_local_variable(
            'index', Int64TensorType())
        distance_variable = scope.declare_local_variable(
            'distance', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(index_variable)
        this_operator.outputs.append(distance_variable)

    elif type(model) in {GaussianMixture, BayesianGaussianMixture}:
        label_variable = scope.declare_local_variable(
            'label', Int64TensorType())
        prob_variable = scope.declare_local_variable(
            'probabilities', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(prob_variable)
        options = scope.get_options(model, dict(score_samples=False))
        if options['score_samples']:
            scores_var = scope.declare_local_variable(
                'score_samples', guess_tensor_type(inputs[0].type))
            this_operator.outputs.append(scores_var)
    else:
        # We assume that all scikit-learn operator produce a single output.
        variable = scope.declare_local_variable(
            'variable', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(variable)

    options = scope.get_options(model, dict(decision_path=False), fail=False)
    if options is not None and options['decision_path']:
        dec_path = scope.declare_local_variable(
            'decision_path', StringTensorType())
        this_operator.outputs.append(dec_path)

    return this_operator.outputs


def _parse_sklearn_pipeline(scope, model, inputs, custom_parsers=None):
    """
    The basic ideas of scikit-learn parsing:
        1. Sequentially go though all stages defined in the considered
           scikit-learn pipeline
        2. The output variables of one stage will be fed into its next
           stage as the inputs.

    :param scope: Scope object defined in _topology.py
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    """
    for step in model.steps:
        inputs = _parse_sklearn(scope, step[1], inputs,
                                custom_parsers=custom_parsers)
    return inputs


def _parse_sklearn_feature_union(scope, model, inputs, custom_parsers=None):
    """
    :param scope: Scope object
    :param model: A scikit-learn FeatureUnion object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by feature union
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, transform in model.transformer_list:
        transformed_result_names.append(
            _parse_sklearn_simple_model(
                scope, transform, inputs,
                custom_parsers=custom_parsers)[0])
        if (model.transformer_weights is not None and name in
                model.transformer_weights):
            transform_result = [transformed_result_names.pop()]
            # Create a Multiply ONNX node
            multiply_operator = scope.declare_local_operator('SklearnMultiply')
            multiply_operator.inputs = transform_result
            multiply_operator.operand = model.transformer_weights[name]
            multiply_output = scope.declare_local_variable(
                'multiply_output', guess_tensor_type(inputs[0].type))
            multiply_operator.outputs.append(multiply_output)
            transformed_result_names.append(multiply_operator.outputs[0])

    # Create a Concat ONNX node
    concat_operator = scope.declare_local_operator('SklearnConcat')
    concat_operator.inputs = transformed_result_names

    # Declare output name of scikit-learn FeatureUnion
    union_name = scope.declare_local_variable(
        'union', guess_tensor_type(inputs[0].type))
    concat_operator.outputs.append(union_name)

    return concat_operator.outputs


def _parse_sklearn_column_transformer(scope, model, inputs,
                                      custom_parsers=None):
    """
    :param scope: Scope object
    :param model: A *scikit-learn* *ColumnTransformer* object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by column transformer
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, op, column_indices in model.transformers_:
        if op == 'drop':
            continue
        if isinstance(column_indices, slice):
            column_indices = list(range(
                column_indices.start
                if column_indices.start is not None else 0,
                column_indices.stop, column_indices.step
                if column_indices.step is not None else 1))
        elif isinstance(column_indices, (int, str)):
            column_indices = [column_indices]
        names = get_column_indices(column_indices, inputs, multiple=True)
        transform_inputs = []
        for onnx_var, onnx_is in names.items():
            tr_inputs = _fetch_input_slice(scope, [inputs[onnx_var]], onnx_is)
            transform_inputs.extend(tr_inputs)

        merged_cols = False
        if len(transform_inputs) > 1:
            if isinstance(op, Pipeline):
                if not isinstance(op.steps[0][1], do_not_merge_columns):
                    merged_cols = True
            elif not isinstance(op, do_not_merge_columns):
                merged_cols = True

        if merged_cols:
            # Many ONNX operators expect one input vector,
            # the default behaviour is to merge columns.
            ty = transform_inputs[0].type.__class__([None, None])

            conc_op = scope.declare_local_operator('SklearnConcat')
            conc_op.inputs = transform_inputs
            conc_names = scope.declare_local_variable('merged_columns', ty)
            conc_op.outputs.append(conc_names)
            transform_inputs = [conc_names]

        model_obj = model.named_transformers_[name]
        if isinstance(model_obj, str):
            if model_obj == "passthrough":
                var_out = transform_inputs[0]
            elif model_obj == "drop":
                var_out = None
            else:
                raise RuntimeError("Unknown operator alias "
                                   "'{0}'. These are specified in "
                                   "_supported_operators.py."
                                   "".format(model_obj))
        else:
            var_out = _parse_sklearn(
                scope, model_obj,
                transform_inputs, custom_parsers=custom_parsers)[0]
            if (model.transformer_weights is not None and name in
                    model.transformer_weights):
                # Create a Multiply ONNX node
                multiply_operator = scope.declare_local_operator(
                    'SklearnMultiply')
                multiply_operator.inputs.append(var_out)
                multiply_operator.operand = model.transformer_weights[name]
                var_out = scope.declare_local_variable(
                    'multiply_output', guess_tensor_type(inputs[0].type))
                multiply_operator.outputs.append(var_out)
        if var_out:
            transformed_result_names.append(var_out)

    # Create a Concat ONNX node
    if len(transformed_result_names) > 1:
        ty = transformed_result_names[0].type.__class__([None, None])
        concat_operator = scope.declare_local_operator('SklearnConcat')
        concat_operator.inputs = transformed_result_names

        # Declare output name of scikit-learn ColumnTransformer
        transformed_column_name = scope.declare_local_variable(
            'transformed_column', ty)
        concat_operator.outputs.append(transformed_column_name)
        return concat_operator.outputs
    return transformed_result_names


def _parse_sklearn_grid_search_cv(scope, model, inputs, custom_parsers=None):
    return (_parse_sklearn_classifier(
                scope, model, inputs, custom_parsers=None)
            if is_classifier(model) else
            _parse_sklearn_simple_model(scope, model, inputs,
                                        custom_parsers=custom_parsers))


def _parse_sklearn_classifier(scope, model, inputs, custom_parsers=None):
    probability_tensor = _parse_sklearn_simple_model(
            scope, model, inputs, custom_parsers=custom_parsers)
    if model.__class__ in [NuSVC, SVC] and not model.probability:
        return probability_tensor
    options = scope.get_options(model, dict(zipmap=True))
    if isinstance(options['zipmap'], bool) and not options['zipmap']:
        return probability_tensor

    if options['zipmap'] == 'columns':
        this_operator = scope.declare_local_operator('SklearnZipMapColumns')
        classes = get_label_classes(scope, model)
        classes_names = get_label_classes(scope, model, node_names=True)
    else:
        this_operator = scope.declare_local_operator('SklearnZipMap')
        classes = get_label_classes(scope, model)

    this_operator.inputs = probability_tensor
    label_type = Int64TensorType([None])

    if (isinstance(model.classes_, list) and
            isinstance(model.classes_[0], np.ndarray)):
        # multi-label problem
        pass
    elif np.issubdtype(classes.dtype, np.floating):
        classes = np.array(list(map(lambda x: int(x), classes)))
        if set(map(lambda x: float(x), classes)) != set(model.classes_):
            raise RuntimeError("skl2onnx implicitly converts float class "
                               "labels into integers but at least one label "
                               "is not an integer. Class labels should "
                               "be integers or strings.")
        this_operator.classlabels_int64s = classes
    elif np.issubdtype(classes.dtype, np.signedinteger):
        this_operator.classlabels_int64s = classes
    elif np.issubdtype(classes.dtype, np.unsignedinteger):
        this_operator.classlabels_int64s = classes
    else:
        classes = np.array([s.encode('utf-8') for s in classes])
        this_operator.classlabels_strings = classes
        label_type = StringTensorType([None])

    output_label = scope.declare_local_variable('output_label', label_type)
    this_operator.outputs.append(output_label)

    if options['zipmap'] == 'columns':
        prob_type = probability_tensor[1].type
        for cl in classes_names:
            output_cl = scope.declare_local_variable(cl, prob_type.__class__())
            this_operator.outputs.append(output_cl)
    else:
        output_probability = scope.declare_local_variable(
            'output_probability',
            SequenceType(
                DictionaryType(
                    label_type, guess_tensor_type(inputs[0].type))))
        this_operator.outputs.append(output_probability)
    return this_operator.outputs


def _parse_sklearn_gaussian_process(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(
            model, dict(return_cov=False, return_std=False))
    if options['return_std'] and options['return_cov']:
        raise RuntimeError(
            "Not returning standard deviation of predictions when "
            "returning full covariance.")

    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    mean_tensor = scope.declare_local_variable(
        "GPmean", guess_tensor_type(inputs[0].type))
    this_operator.inputs = inputs
    this_operator.outputs.append(mean_tensor)

    if options['return_std'] or options['return_cov']:
        # covariance or standard deviation
        covstd_tensor = scope.declare_local_variable(
            'GPcovstd', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(covstd_tensor)
    return this_operator.outputs


def _parse_sklearn_bayesian_ridge(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(return_std=False))
    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    mean_tensor = scope.declare_local_variable(
        "variable", guess_tensor_type(inputs[0].type))
    this_operator.inputs = inputs
    this_operator.outputs.append(mean_tensor)

    if options['return_std']:
        # covariance or standard deviation
        covstd_tensor = scope.declare_local_variable(
            'std', guess_tensor_type(inputs[0].type))
        this_operator.outputs.append(covstd_tensor)
    return this_operator.outputs


def _parse_sklearn(scope, model, inputs, custom_parsers=None,
                   final_types=None):
    """
    This is a delegate function. It does nothing but invokes the
    correct parsing function according to the input model's type.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder
        and LogisticRegression)
    :param inputs: A list of variables
    :param custom_parsers: parsers determines which outputs is expected
        for which particular task, default parsers are defined for
        classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary ``{ type: fct_parser(scope,
        model, inputs, custom_parsers=None) }``
    :param final_types: a python list. Works the same way as initial_types
        but not mandatory, it is used to overwrites the type
        (if type is not None) and the name of every output.
    :return: The output variables produced by the input model
    """
    if final_types is not None:
        outputs = []
        for name, ty in final_types:
            var = scope.declare_local_variable(name, ty)
            if var.onnx_name != name:
                raise RuntimeError(
                    "Unable to add duplicated output '{}', '{}'.".format(
                        var.onnx_name, name))
            outputs.append(var)
        hidden_outputs = _parse_sklearn(
            scope, model, inputs, custom_parsers=custom_parsers)
        if len(hidden_outputs) != len(outputs):
            raise RuntimeError(
                "Number of declared outputs is unexpected, declared '{}' "
                "found '{}'.".format(
                    ", ".join(_.onnx_name for _ in outputs),
                    ", ".join(_.onnx_name for _ in hidden_outputs)))
        for h, o in zip(hidden_outputs, outputs):
            if o.type is None:
                iop = scope.declare_local_operator('SklearnIdentity')
            else:
                iop = scope.declare_local_operator('SklearnCast')
            iop.inputs = [h]
            iop.outputs = [o]
        return outputs

    tmodel = type(model)
    if custom_parsers is not None and tmodel in custom_parsers:
        outputs = custom_parsers[tmodel](scope, model, inputs,
                                         custom_parsers=custom_parsers)
    elif tmodel in sklearn_parsers_map:
        outputs = sklearn_parsers_map[tmodel](scope, model, inputs,
                                              custom_parsers=custom_parsers)
    elif isinstance(model, pipeline.Pipeline):
        parser = sklearn_parsers_map[pipeline.Pipeline]
        outputs = parser(scope, model, inputs, custom_parsers=custom_parsers)
    else:
        outputs = _parse_sklearn_simple_model(scope, model, inputs,
                                              custom_parsers=custom_parsers)
    return outputs


def parse_sklearn(scope, model, inputs, custom_parsers=None, final_types=None):
    """
    This is a delegate function. It does nothing but invokes the
    correct parsing function according to the input model's type.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder
        and LogisticRegression)
    :param inputs: A list of variables
    :param custom_parsers: parsers determines which outputs is expected
        for which particular task, default parsers are defined for
        classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary ``{ type: fct_parser(scope,
        model, inputs, custom_parsers=None) }``
    :param final_types: a python list. Works the same way as initial_types
        but not mandatory, it is used to overwrites the type
        (if type is not None) and the name of every output.
    :return: The output variables produced by the input model
    """
    if final_types is None and has_pipeline(model):
        try:
            outputs = _parse_sklearn(
                scope.temp(), model, inputs, custom_parsers=custom_parsers)
        except RuntimeError:
            return _parse_sklearn(
                scope, model, inputs, custom_parsers=custom_parsers)

        reserved = []
        for o in outputs:
            reserved.append(scope.reserve_name(o.raw_name))
    else:
        reserved = None

    res = _parse_sklearn(
        scope, model, inputs, custom_parsers=custom_parsers,
        final_types=final_types)

    if final_types is None and reserved is not None:
        for r in reserved:
            scope.unreserve_name(r)

        outputs = []
        for var, name in zip(res, reserved):
            var2 = scope.declare_local_variable(name, var.type)
            outputs.append(var2)

        for h, o in zip(res, outputs):
            iop = scope.declare_local_operator('SklearnIdentity')
            iop.inputs = [h]
            iop.outputs = [o]

        return outputs

    return res


def parse_sklearn_model(model, initial_types=None, target_opset=None,
                        custom_conversion_functions=None,
                        custom_shape_calculators=None,
                        custom_parsers=None,
                        options=None, white_op=None,
                        black_op=None, final_types=None):
    """
    Puts *scikit-learn* object into an abstract container so that
    our framework can work seamlessly on models created
    with different machine learning tools.

    :param model: A scikit-learn model
    :param initial_types: a python list. Each element is a tuple of a
        variable name and a type defined in data_types.py
    :param target_opset: number, for example, 7 for ONNX 1.2,
        and 8 for ONNX 1.3.
    :param custom_conversion_functions: a dictionary for specifying
        the user customized conversion function if not registered
    :param custom_shape_calculators: a dictionary for specifying the
        user customized shape calculator if not registered
    :param custom_parsers: parsers determines which outputs is expected
        for which particular task, default parsers are defined for
        classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary
        ``{ type: fct_parser(scope, model, inputs, custom_parsers=None) }``
    :param options: specific options given to converters
        (see :ref:`l-conv-options`)
    :param white_op: white list of ONNX nodes allowed
        while converting a pipeline, if empty, all are allowed
    :param black_op: black list of ONNX nodes allowed
        while converting a pipeline, if empty, none are blacklisted
    :param final_types: a python list. Works the same way as initial_types
        but not mandatory, it is used to overwrites the type
        (if type is not None) and the name of every output.
    :return: :class:`Topology <skl2onnx.common._topology.Topology>`
    """
    options = _process_options(model, options)

    raw_model_container = SklearnModelContainerNode(
        model, white_op=white_op, black_op=black_op)

    # Declare a computational graph. It will become a representation of
    # the input scikit-learn model after parsing.
    topology = Topology(
            raw_model_container, initial_types=initial_types,
            target_opset=target_opset,
            custom_conversion_functions=custom_conversion_functions,
            custom_shape_calculators=custom_shape_calculators,
            registered_models=dict(
                conv=_converter_pool, shape=_shape_calculator_pool,
                aliases=sklearn_operator_name_map))

    # Declare an object to provide variables' and operators' naming mechanism.
    # In contrast to CoreML, one global scope
    # is enough for parsing scikit-learn models.
    scope = topology.declare_scope('__root__', options=options)

    # Declare input variables. They should be the inputs of the scikit-learn
    # model you want to convert into ONNX.
    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    # The object raw_model_container is a part of the topology
    # we're going to return. We use it to store the inputs of
    # the scikit-learn's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input scikit-learn model as a Topology object.
    outputs = parse_sklearn(scope, model, inputs,
                            custom_parsers=custom_parsers,
                            final_types=final_types)

    # The object raw_model_container is a part of the topology we're
    # going to return. We use it to store the outputs of the
    # scikit-learn's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology


def build_sklearn_parsers_map():
    map_parser = {
        pipeline.Pipeline: _parse_sklearn_pipeline,
        pipeline.FeatureUnion: _parse_sklearn_feature_union,
        BayesianRidge: _parse_sklearn_bayesian_ridge,
        GaussianProcessRegressor: _parse_sklearn_gaussian_process,
        GridSearchCV: _parse_sklearn_grid_search_cv,
    }
    if ColumnTransformer is not None:
        map_parser[ColumnTransformer] = _parse_sklearn_column_transformer

    for tmodel in sklearn_classifier_list:
        if tmodel not in [LinearSVC]:
            map_parser[tmodel] = _parse_sklearn_classifier
    return map_parser


def update_registered_parser(model, parser_fct):
    """
    Registers or updates a parser for a new model.
    A parser returns the expected output of a model.

    :param model: model class
    :param parser_fct: parser, signature is the same as
        :func:`parse_sklearn <skl2onnx._parse.parse_sklearn>`
    """
    check_signature(parser_fct, _parse_sklearn_classifier)
    sklearn_parsers_map[model] = parser_fct


# registered parsers
sklearn_parsers_map = build_sklearn_parsers_map()
