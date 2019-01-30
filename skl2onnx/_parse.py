# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import warnings
import numpy as np

from .common._container import SklearnModelContainer
from .common._topology import Topology, Variable, Operator, Scope, convert_topology
from .common._registration import register_converter, register_shape_calculator, get_shape_calculator
from .common.data_types import DataType, Int64Type, FloatType, StringType, TensorType, find_type_conversion
from .common.data_types import FloatTensorType, Int64TensorType, SequenceType, DictionaryType
from .common.utils import get_column_indices
from .shape_calculators.Concat import calculate_sklearn_concat

# Pipeline
from sklearn import pipeline
from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin

# Calibrated classifier CV
from sklearn.calibration import CalibratedClassifierCV

# Column Transformer
from sklearn.compose import ColumnTransformer

# Linear classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

# Linear regressors
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR

# Tree-based models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# Support vector machines
from sklearn.svm import SVC, SVR, NuSVC, NuSVR

# K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors

# Naive Bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

# Clustering
from sklearn.cluster import KMeans, MiniBatchKMeans

# Operators for preprocessing and feature engineering
from sklearn.decomposition import PCA 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import GenericUnivariateSelect, RFE, RFECV, SelectFdr, SelectFpr, SelectFromModel
from sklearn.feature_selection import SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.impute import SimpleImputer

# In most cases, scikit-learn operator produces only one output. However, each classifier has basically two outputs;
# one is the predicted label and the other one is the probabilities of all possible labels. Here is a list of supported
# scikit-learn classifiers. In the parsing stage, we produce two outputs for objects included in the following list and
# one output for everything not in the list.
sklearn_classifier_list = [LogisticRegression, SGDClassifier, LinearSVC, SVC, NuSVC,
                           GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier,
                           ExtraTreesClassifier, BernoulliNB, MultinomialNB, KNeighborsClassifier,
                           CalibratedClassifierCV]

# Clustering algorithms: produces two outputs, label and score for each cluster in most cases.
cluster_list = [KMeans, MiniBatchKMeans]

# Associate scikit-learn types with our operator names. If two scikit-learn models share a single name, it means their
# are equivalent in terms of conversion.

def build_sklearn_operator_name_map():
    res = {k: "Sklearn" + k.__name__ for k in [
                    RobustScaler, LinearSVC, OneHotEncoder, DictVectorizer, Imputer, SimpleImputer,
                    LabelBinarizer, LabelEncoder, SVC, SVR, LinearSVR, LinearRegression, Lasso,
                    LassoLars, Ridge, Normalizer, DecisionTreeClassifier, DecisionTreeRegressor,
                    RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
                    ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                    CalibratedClassifierCV, KNeighborsClassifier, KNeighborsRegressor,
                    NearestNeighbors, MultinomialNB, BernoulliNB, KMeans, MiniBatchKMeans,
                    Binarizer, PCA, TruncatedSVD, MinMaxScaler, MaxAbsScaler,
                    CountVectorizer, TfidfVectorizer, TfidfTransformer,
                    GenericUnivariateSelect, RFE, RFECV, SelectFdr, SelectFpr, SelectFromModel,
                    SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold,
                    FunctionTransformer]}
    res.update({
        ElasticNet: 'SklearnElasticNetRegressor',
        LinearRegression: 'SklearnLinearRegressor',
        LogisticRegression: 'SklearnLinearClassifier',
        NuSVC: 'SklearnSVC',
        NuSVR: 'SklearnSVR',
        SGDClassifier: 'SklearnLinearClassifier',
        SGDRegressor: 'SklearnLinearRegressor',
        StandardScaler: 'SklearnScaler',
    })
    return res


sklearn_operator_name_map = build_sklearn_operator_name_map()


def update_registered_converter(model, alias, shape_fct, convert_fct, overwrite=True):
    """
    Registers or updates a converter for a new model so that
    it can be converted when inserted in a *scikit-learn* pipeline.
    
    :param model: model class
    :param alias: alias used to register the model
    :param shape_fct: function which checks or modifies the expected outputs,
        this function should be fast so that the whole graph can be computed followed
        by the conversion of each model, parallelized or not
    :param convert_fct: function which converts a model
    :param overwrite: False to raise exception if a converter already exists

    The alias is usually the library name followed by the model name.
    Example:

    ::

        from onnxmltools.convert.common.shape_calculator import calculate_linear_classifier_output_shapes
        from skl2onnx.operator_converters.RandomForest import convert_sklearn_random_forest_classifier
        from skl2onnx import update_registered_converter
        update_registered_converter(SGDClassifier, 'SklearnLinearClassifier',
                                    calculate_linear_classifier_output_shapes,
                                    convert_sklearn_random_forest_classifier)
    """    
    if not overwrite and model in sklearn_operator_name_map and alias != sklearn_operator_name_map[model]:
        warnings.warn("Model '{0}' was already registered under alias '{1}'.".format(
            model, sklearn_operator_name_map[model]))
    sklearn_operator_name_map[model] = alias
    register_converter(alias, convert_fct, overwrite=overwrite)
    register_shape_calculator(alias, shape_fct, overwrite=overwrite)


def _get_sklearn_operator_name(model_type):
    '''
    Get operator name of the input argument

    :param model_type:  A scikit-learn object (e.g., SGDClassifier and Binarizer)
    :return: A string which stands for the type of the input model in our conversion framework
    '''
    if model_type not in sklearn_operator_name_map:
        raise ValueError("No proper operator name found for '%s'" % model_type)
    return sklearn_operator_name_map[model_type]


def _parse_sklearn_simple_model(scope, model, inputs):
    '''
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    '''
    this_operator = scope.declare_local_operator(_get_sklearn_operator_name(type(model)), model)
    this_operator.inputs = inputs

    if type(model) in sklearn_classifier_list or isinstance(model, ClassifierMixin):
        # For classifiers, we may have two outputs, one for label and the other one for probabilities of all classes.
        # Notice that their types here are not necessarily correct and they will be fixed in shape inference phase
        label_variable = scope.declare_local_variable('label', FloatTensorType())
        probability_tensor_variable = scope.declare_local_variable('probabilities', FloatTensorType())
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_tensor_variable)
    elif type(model) in cluster_list or isinstance(model, ClusterMixin):
        # For clustering, we may have two outputs, one for label and the other one for scores of all classes.
        # Notice that their types here are not necessarily correct and they will be fixed in shape inference phase
        label_variable = scope.declare_local_variable('label', Int64TensorType())
        score_tensor_variable = scope.declare_local_variable('scores', FloatTensorType())
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(score_tensor_variable)
    elif type(model) == NearestNeighbors:
        # For Nearest Neighbours, we have two outputs, one for nearest neighbours' indices
        # and the other one for distances
        index_variable = scope.declare_local_variable('index', Int64TensorType())
        distance_variable = scope.declare_local_variable('distance', FloatTensorType())
        this_operator.outputs.append(index_variable)
        this_operator.outputs.append(distance_variable)
    else:
        # We assume that all scikit-learn operator produce a single output.
        variable = scope.declare_local_variable('variable', FloatTensorType())
        this_operator.outputs.append(variable)
        
    # We call the shape calculator.
    name = sklearn_operator_name_map[type(model)]
    shape_calc = get_shape_calculator(name)
    try:
        shape_calc(this_operator)
    except RuntimeError as e:
        inps = "\n".join(str(v) for v in inputs)
        raise RuntimeError("Unable to precise output types for '{0}' due to: {1}.\nInputs:\n{2}\nFunction:{3}".format(type(model), e, inps, shape_calc))
    return this_operator.outputs


def _parse_sklearn_pipeline(scope, model, inputs):
    '''
    The basic ideas of scikit-learn parsing:
        1. Sequentially go though all stages defined in the considered scikit-learn pipeline
        2. The output variables of one stage will be fed into its next stage as the inputs.

    :param scope: Scope object defined in _topology.py
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    '''
    for step in model.steps:
        inputs = _parse_sklearn(scope, step[1], inputs)
    return inputs


def _parse_sklearn_feature_union(scope, model, inputs):
    '''
    :param scope: Scope object
    :param model: A scikit-learn FeatureUnion object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by feature union
    '''
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, transform in model.transformer_list:
        transformed_result_names.append(_parse_sklearn_simple_model(scope, transform, inputs)[0])
    # Create a Concat ONNX node
    concat_operator = scope.declare_local_operator('SklearnConcat')
    concat_operator.inputs = transformed_result_names

    # Declare output name of scikit-learn FeatureUnion
    union_name = scope.declare_local_variable('union', FloatTensorType())
    concat_operator.outputs.append(union_name)
    calculate_sklearn_concat(concat_operator)

    return concat_operator.outputs


def _fetch_input_slice(scope, inputs, column_indices):
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list of 1 input.")
    if len(inputs) == 0:
        raise RuntimeError("Operator ArrayFeatureExtractor requires at least one inputs.")
    if len(inputs) != 1:
        raise RuntimeError("Operator ArrayFeatureExtractor does not support multiple input tensors.")
    if isinstance(inputs[0].type, TensorType) and inputs[0].type.shape[1] == len(column_indices):
        # No need to extract.
        return inputs
    array_feature_extractor_operator = scope.declare_local_operator('SklearnArrayFeatureExtractor')
    array_feature_extractor_operator.inputs = inputs
    array_feature_extractor_operator.column_indices = column_indices
    output_variable_name = scope.declare_local_variable('extracted_feature_columns', inputs[0].type)
    array_feature_extractor_operator.outputs.append(output_variable_name)
    return array_feature_extractor_operator.outputs


def _parse_sklearn_column_transformer(scope, model, inputs):
    '''
    :param scope: Scope object
    :param model: A scikit-learn ColumnTransformer object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by column transformer
    '''
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, transform, column_indices in model.transformers:
        if isinstance(column_indices, slice):
            column_indices = list(range(column_indices.start if column_indices.start is not None else 0,
                                        column_indices.stop, column_indices.step if column_indices.step
                                        is not None else 1))
        names = get_column_indices(column_indices, inputs, multiple=True)
        transform_inputs = []
        for onnx_var, onnx_is in names.items():
            tr_inputs = _fetch_input_slice(scope, [inputs[onnx_var]], onnx_is)
            transform_inputs.extend(tr_inputs)
        
        if len(transform_inputs) > 1:            
            # Many ONNX operators expect one input vector,
            # the default behviour is to merge columns.
            nb_cols = sum(inp.type.shape[1] for inp in transform_inputs)
            ty = transform_inputs[0].type.__class__([1, nb_cols])
            
            concop = scope.declare_local_operator('SklearnConcat')
            concop.inputs = transform_inputs
            concnames = scope.declare_local_variable('merged_columns', ty)
            concop.outputs.append(concnames)
            transform_inputs = [concnames]
            calculate_sklearn_concat(concop)
        
        var_out = _parse_sklearn(scope, model.named_transformers_[name], transform_inputs)[0]
        transformed_result_names.append(var_out)

    # Create a Concat ONNX node
    if len(transformed_result_names) > 1:
        nb_cols = sum(inp.type.shape[1] for inp in transformed_result_names)
        ty = transformed_result_names[0].type.__class__([1, nb_cols])
        concat_operator = scope.declare_local_operator('SklearnConcat')
        concat_operator.inputs = transformed_result_names

        # Declare output name of scikit-learn ColumnTransformer
        transformed_column_name = scope.declare_local_variable('transformed_column', ty)
        concat_operator.outputs.append(transformed_column_name)
        calculate_sklearn_concat(concat_operator)

        return concat_operator.outputs
    else:
        return transformed_result_names


def _parse_sklearn(scope, model, inputs):
    '''
    This is a delegate function. It doesn't nothing but invoke the correct parsing function according to the input
    model's type.
    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    
    TODO: Output should be created in each shape calculator.
    '''
    if isinstance(model, pipeline.Pipeline):
        outputs = _parse_sklearn_pipeline(scope, model, inputs)
    elif isinstance(model, pipeline.FeatureUnion):
        outputs = _parse_sklearn_feature_union(scope, model, inputs)
    elif isinstance(model, ColumnTransformer):
        outputs = _parse_sklearn_column_transformer(scope, model, inputs)
    elif type(model) in sklearn_classifier_list and type(model) not in [LinearSVC, SVC, NuSVC]:
        probability_tensor = _parse_sklearn_simple_model(scope, model, inputs)
        this_operator = scope.declare_local_operator('SklearnZipMap')
        this_operator.inputs = probability_tensor
        classes = model.classes_
        label_type = Int64Type()

        if np.issubdtype(model.classes_.dtype, np.floating):
            classes = np.array(list(map(lambda x: int(x), classes)))
            this_operator.classlabels_int64s = classes
        elif np.issubdtype(model.classes_.dtype, np.signedinteger):
            this_operator.classlabels_int64s = classes
        else:
            classes = np.array([s.encode('utf-8') for s in classes])
            this_operator.classlabels_strings = classes
            label_type = StringType()
        output_label = scope.declare_local_variable('output_label', label_type)
        output_probability = scope.declare_local_variable('output_probability',
                             SequenceType(DictionaryType(label_type, FloatTensorType())))
        this_operator.outputs.append(output_label)
        this_operator.outputs.append(output_probability)
        outputs = this_operator.outputs
    else:
        outputs = _parse_sklearn_simple_model(scope, model, inputs)
        
    # scikit-learn does not implements models without unknown output shapes.
    for i, o in enumerate(outputs):
        if isinstance(o.type, TensorType) and (o.type.shape is None or \
                'None' in o.type.shape or None in o.type.shape):
            raise RuntimeError("No output can have unkown shape.\nOutput {0}={1}\nModel={2}".format(i, o, model))
    return outputs


def parse_sklearn(model, initial_types=None, target_opset=None,
                  custom_conversion_functions=None, custom_shape_calculators=None):
    # Put scikit-learn object into an abstract container so that our framework can work seamlessly on models created
    # with different machine learning tools.
    raw_model_container = SklearnModelContainer(model)

    # Declare a computational graph. It will become a representation of the input scikit-learn model after parsing.
    topology = Topology(raw_model_container,
                        initial_types=initial_types,
                        target_opset=target_opset,
                        custom_conversion_functions=custom_conversion_functions,
                        custom_shape_calculators=custom_shape_calculators)

    # Declare an object to provide variables' and operators' naming mechanism. In contrast to CoreML, one global scope
    # is enough for parsing scikit-learn models.
    scope = topology.declare_scope('__root__')

    # Declare input variables. They should be the inputs of the scikit-learn model you want to convert into ONNX
    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    # The object raw_model_container is a part of the topology we're going to return. We use it to store the inputs of
    # the scikit-learn's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input scikit-learn model as a Topology object.
    outputs = _parse_sklearn(scope, model, inputs)

    # THe object raw_model_container is a part of the topology we're going to return. We use it to store the outputs of
    # the scikit-learn's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology
