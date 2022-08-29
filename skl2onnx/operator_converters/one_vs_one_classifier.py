# SPDX-License-Identifier: Apache-2.0

from sklearn.base import is_regressor
from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_concat, apply_identity, apply_mul, apply_reshape, apply_transpose)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_normalization
from ..common._apply_operation import (
    apply_slice, apply_sub, apply_cast, apply_abs, apply_add, apply_div)
from ..common.utils_classifier import _finalize_converter_classes
from ..common.data_types import guess_proto_type, Int64TensorType, FloatTensorType
from .._supported_operators import sklearn_operator_name_map


def convert_one_vs_one_classifier(scope: Scope, operator: Operator,
                                   container: ModelComponentContainer):

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator
    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options['raw_scores']
    label_names = []
    score_names = []
    for i, estimator in enumerate(op.estimators_):
        op_type = sklearn_operator_name_map[type(estimator)]
        label_name_out = 'label_out_%d' % i
        score_name_out = 'score_out_%d' % i
        if op_type == 'SklearnLinearSVC':
            attrs = {
                'coefficients': estimator.coef_.ravel().tolist(),
                'intercepts': estimator.intercept_.ravel().tolist()
            }
            container.add_node(
                op_type = 'LinearClassifier', 
                inputs = operator.inputs[0].full_name,
                outputs = [label_name_out, score_name_out],
                name = 'LinearClassifier_%d' % i, 
                op_domain='ai.onnx.ml', **attrs
            )

        label_names.append(label_name_out)
        score_names.append(score_name_out)
    # end for

    # label
    apply_concat(scope, label_names, 'concat_label_out', container, axis=0)

    container.add_initializer('const_1', onnx_proto.TensorProto.FLOAT, [], [1])
    container.add_initializer('const_3', onnx_proto.TensorProto.FLOAT, [], [3])
    container.add_initializer(
        'target_shape', onnx_proto.TensorProto.INT64, 
        [2], [op.n_classes_, -1])

    apply_reshape(
        scope, 'concat_label_out', 'reshape_label_out',
        container, desired_shape='target_shape')

    apply_transpose(
        scope, 'reshape_label_out', 'transpose_label_out', 
        container, perm=(1, 0))

    apply_cast(
        scope, 'transpose_label_out', 'cast_label_out', 
        container, to=onnx_proto.TensorProto.FLOAT
    )

    # score
    apply_concat(
        scope, score_names, 'concat_score_out',
        container, 'concat_score', axis=0)

    apply_reshape(
        scope, 'concat_score_out', 'reshape_score_out',
        container, desired_shape='target_shape')

    apply_transpose(
        scope, 'reshape_score_out', 'transpose_score_out',
        container, perm=(1, 0))

    n_classes = len(op.classes_)
    sumc_name = [scope.get_unique_variable_name('svcsumc_%d' % i)
                 for i in range(n_classes)]
    vote_name = [scope.get_unique_variable_name('svcvote_%d' % i)
                 for i in range(n_classes)]
    sumc_add = {n: [] for n in sumc_name}
    vote_add = {n: [] for n in vote_name}
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # confidences
            name = scope.get_unique_operator_name(
                'ArrayFeatureExtractor')
            ext = scope.get_unique_variable_name('Csvc_%d' % k)
            ind = scope.get_unique_variable_name('Cind_%d' % k)
            container.add_initializer(
                ind, onnx_proto.TensorProto.INT64, [1], [k])
            container.add_node(
                'ArrayFeatureExtractor', ['transpose_score_out', ind],
                ext, op_domain='ai.onnx.ml', name=name)
            sumc_add[sumc_name[i]].append(ext)

            neg = scope.get_unique_variable_name('Cneg_%d' % k)
            name = scope.get_unique_operator_name('Neg')
            container.add_node(
                'Neg', ext, neg, op_domain='', name=name,
                op_version=6)
            sumc_add[sumc_name[j]].append(neg)

            # votes
            ext = scope.get_unique_variable_name('Vsvcv_%d' % k)
            container.add_node('ArrayFeatureExtractor', 
                               ['cast_label_out', ind],
                               ext, op_domain='ai.onnx.ml')
            vote_add[vote_name[j]].append(ext)

            neg = scope.get_unique_variable_name('Vnegv_%d' % k)
            container.add_node('Neg', ext, neg)
            
            neg1 = scope.get_unique_variable_name('Vnegv1_%d' % k)
            apply_add(scope, [neg, 'const_1'], neg1, container, broadcast=1,
                      operator_name='AddCl_%d_%d' % (i, j))
            vote_add[vote_name[i]].append(neg1)

            # next
            k += 1

    for k, v in sumc_add.items():
        name = scope.get_unique_operator_name('Sum')
        container.add_node(
            'Sum', v, k, op_domain='', name=name, op_version=8)
    for k, v in vote_add.items():
        name = scope.get_unique_operator_name('Sum')
        container.add_node(
            'Sum', v, k, op_domain='', name=name, op_version=8)

    conc = scope.get_unique_variable_name('Csvcconc')
    apply_concat(scope, sumc_name, conc, container, axis=1)
    conc_vote = scope.get_unique_variable_name('Vsvcconcv')
    apply_concat(scope, vote_name, conc_vote, container, axis=1)

    conc_abs = scope.get_unique_variable_name('Cabs')
    apply_abs(scope, conc, conc_abs, container)


    conc_abs1 = scope.get_unique_variable_name('Cconc_abs1')
    apply_add(scope, [conc_abs, 'const_1'], conc_abs1, container, broadcast=1,
              operator_name='AddF0')
    conc_abs3 = scope.get_unique_variable_name('Cconc_abs3')
    apply_mul(scope, [conc_abs1, 'const_3'], conc_abs3, container, broadcast=1)

    final = scope.get_unique_variable_name('Csvcfinal')
    apply_div(
        scope, [conc, conc_abs3], final, container, broadcast=0)

    output_name = operator.outputs[1].full_name
    apply_add(
        scope, [conc_vote, final], output_name, container, broadcast=0,
        operator_name='AddF1')

    container.add_node('ArgMax', output_name, operator.outputs[0].full_name,
                       axis=1)


register_converter('SklearnOneVsOneClassifier',
                   convert_one_vs_one_classifier,
                   options={'zipmap': [True, False, 'columns'],
                            'nocl': [True, False],
                            'output_class_labels': [False, True],
                            'raw_scores': [True, False]})

