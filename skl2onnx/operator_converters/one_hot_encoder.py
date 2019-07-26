# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import collections
import numpy as np
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common.utils import get_column_index, get_column_indices


def convert_sklearn_one_hot_encoder(scope, operator, container):
    op = operator.raw_operator
    C = operator.inputs[0].type.shape[1]

    # encoded_slot_sizes[i] is the number of output coordinates associated
    # with the ith categorical feature
    categorical_values_per_feature = []
    if hasattr(op, 'categories_'):
        categorical_feature_indices = [
                i for i, mat in enumerate(op.categories_)
                if mat is not None and len(mat) > 0]

        for cat in op.categories_:
            if cat is None and len(cat) == 0:
                continue
            all_int = all(map(lambda x: isinstance(
                x, (int, np.int64, np.int32, np.byte)), cat))
            if not all_int:
                all_float = all(map(lambda x: isinstance(
                            x, (float, np.float32, np.float64)), cat))
                if all_float:
                    cat_int = [np.int64(i) for i in cat]
                    if [float(i) for i in cat_int] == [float(i) for i in cat]:
                        cat = np.array(cat_int, dtype=np.int64)
                        all_int = True
            if all_int:
                categorical_values_per_feature.append(
                                list(cat.astype(np.int64)))
            elif cat.dtype in (np.str, np.unicode, np.object):
                categorical_values_per_feature.append([str(_) for _ in cat])
            else:
                raise TypeError("Categories must be int or strings "
                                "not {0}.".format(cat.dtype))
    else:
        # Relies on n_values: deprecated in 0.20,
        # removed in 0.22.
        if op.categorical_features == 'all':
            categorical_feature_indices = [i for i in range(C)]
        elif isinstance(op.categorical_features, collections.Iterable):
            if all(isinstance(i, bool) for i in op.categorical_features):
                categorical_feature_indices = [
                        i for i, active in enumerate(op.categorical_features)
                        if active]
            else:
                categorical_feature_indices = [
                            int(i) for i in op.categorical_features]
        else:
            raise ValueError("Unknown operation mode '{0}'."
                             "".format(op.categorical_features))

        if op.n_values == 'auto':
            # Use active feature to determine output length
            for i in range(len(op.feature_indices_) - 1):
                allowed_values = []
                index_head = op.feature_indices_[i]
                # feature indexed by index_tail not included in this category
                index_tail = op.feature_indices_[i + 1]
                for j in op.active_features_:
                    if index_head <= j and j < index_tail:
                        allowed_values.append(j - index_head)
                categorical_values_per_feature.append(allowed_values)
        elif isinstance(op.n_values, numbers.Integral):
            # Each categorical feature will be mapped to a fixed length
            # one-hot sub-vector
            for i in range(len(op.feature_indices_) - 1):
                index_head = op.feature_indices_[i]
                categorical_values_per_feature.append(
                    list(i - index_head for i in range(op.n_values)))
        else:
            # Each categorical feature has its own sub-vector length
            for max_index in op.n_values:
                categorical_values_per_feature.append(
                        list(i for i in range(max_index)))

    # Variable names produced by one-hot encoders. Each of them is the
    # encoding result of a categorical feature.
    final_variable_names = []
    final_variable_lengths = []
    for i, cats in zip(categorical_feature_indices,
                       categorical_values_per_feature):
        # i is the global column index among the columns processed by this
        # object not the column index inside the input variable for the ONNX
        # graph so basically, i is the index for scikit-learn not ONNX.
        onnx_var, onnx_i = get_column_index(i, operator.inputs)
        onnx_var_name = operator.inputs[onnx_var].full_name

        # Put a feature index we want to encode to a tensor
        index_variable_name = scope.get_unique_variable_name('target_index')
        container.add_initializer(index_variable_name,
                                  onnx_proto.TensorProto.INT64, [1], [onnx_i])

        # Extract the categorical feature from the original input tensor
        extracted_feature_name = scope.get_unique_variable_name(
                'ext_feature_from_%s_at_%d' % (onnx_var_name, onnx_i))
        extractor_type = 'ArrayFeatureExtractor'
        extractor_attrs = {
            'name': scope.get_unique_operator_name(extractor_type)
        }
        container.add_node(
            extractor_type, [onnx_var_name, index_variable_name],
            extracted_feature_name, op_domain='ai.onnx.ml', **extractor_attrs)

        # string or int
        all_int = all(map(
            lambda x: isinstance(
                x, (int, np.int64, np.int32, np.byte)), cats))
        if all_int:
            field = 'cats_int64s'
        else:
            field = "cats_strings"
            cats = [str(c) for c in cats]

        # Encode the extracted categorical feature as a one-hot vector
        encoder_type = 'OneHotEncoder'
        encoder_attrs = {
            'name': scope.get_unique_operator_name(encoder_type),
            field: cats
        }
        encoded_feature_name = scope.get_unique_variable_name(
                                        'encoded_feature_at_%d' % i)
        container.add_node(encoder_type, extracted_feature_name,
                           encoded_feature_name, op_domain='ai.onnx.ml',
                           **encoder_attrs)

        # Collect features produced by one-hot encoder.
        final_variable_names.append(encoded_feature_name)
        # For each categorical value, the length of its encoded result is
        # the number of all possible categorical values.
        final_variable_lengths.append(len(cats))

    # If there are some features which are not processed by one-hot encoding,
    # we extract them directly from the original
    # input and combine them with the outputs of one-hot encoders.
    passed_indices = [i for i in range(C)
                      if i not in categorical_feature_indices]
    if len(passed_indices) > 0:
        variables = get_column_indices(passed_indices, operator.inputs,
                                       multiple=True)
        passed_feature_names = []
        for name, inds in variables.items():
            passed_feature_name = scope.get_unique_variable_name(
                                            'passed_through_features')
            passed_feature_names.append(passed_feature_name)
            extractor_type = 'ArrayFeatureExtractor'
            extractor_attrs = {
                'name': scope.get_unique_operator_name(extractor_type)
            }
            passed_indices_name = scope.get_unique_variable_name(
                                                'passed_feature_indices')
            container.add_initializer(passed_indices_name,
                                      onnx_proto.TensorProto.INT64,
                                      [len(inds)], inds)
            container.add_node(extractor_type,
                               [onnx_var_name, passed_indices_name],
                               passed_feature_name, op_domain='ai.onnx.ml',
                               **extractor_attrs)

        final_variable_names.extend(passed_feature_names)
        final_variable_lengths.append(len(passed_feature_names))

    # Combine encoded features and passed features
    collector_type = 'FeatureVectorizer'
    collector_attrs = {'name': scope.get_unique_operator_name(collector_type)}
    collector_attrs['inputdimensions'] = final_variable_lengths
    container.add_node(collector_type, final_variable_names,
                       operator.outputs[0].full_name,
                       op_domain='ai.onnx.ml', **collector_attrs)


register_converter('SklearnOneHotEncoder', convert_sklearn_one_hot_encoder)
