# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._apply_operation import apply_concat
from ..common._topology import FloatTensorType
from ..common._registration import register_converter
from ..common._apply_operation import apply_normalization
from ..common.utils_classifier import _finalize_converter_classes
from .._supported_operators import sklearn_operator_name_map


def convert_voting_classifier(scope, operator, container):    
    """
    Converts a *VotingClassifier* into *ONNX* format.
    The operator cannot compute mulitple prediction at a time due
    to reduce operators.

    *predict_proba* is not defined by *scikit-learn* when *``voting='hard'``*.
    The converted model still defines a probability vector equal to the
    highest probability obtained for each class over all estimators.

    *scikit-learn* enables both modes, transformer and predictor
    for the voting classifier. *ONNX* does not make this
    distinction and always creates two outputs, labels
    and probabiliries.
    """
    op = operator.raw_operator
    classes = op.classes_

    n_classes = len(op.classes_)

    probs_names = []
    for i, estimator in enumerate(op.estimators_):
        if estimator is None:
            continue
        
        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type)
        this_operator.raw_operator = estimator
        this_operator.inputs = operator.inputs

        label_name = scope.declare_local_variable('label_%d' % i)
        prob_name = scope.declare_local_variable('proba_%d' % i, FloatTensorType())
        this_operator.outputs.append(label_name)
        this_operator.outputs.append(prob_name)

        probs_names.append(prob_name.onnx_name)

    if op.weights is not None:
        raise NotImplementedError("weighted voting is not implemented yet.")

    # concatenates outputs
    conc_name = scope.get_unique_variable_name('concatenated')
    apply_concat(scope, probs_names, [conc_name], container, axis=0)
    
    # aggregation
    if op.voting == 'hard':
        op_name = 'ReduceMax'
    elif op.voting == 'soft':
        op_name = 'ReduceMean'
    else:
        raise RuntimeError("Unuspported voting kind '{}'.".format(op.voting))

    if op.flatten_transform in (False, None):
        red_name = operator.outputs[1].full_name
        container.add_node(op_name, conc_name, red_name,
                           name=scope.get_unique_operator_name(op_name), axes=[0])
    else:
        raise NotImplementedError("flatten_transform==True is not implemented yet.")

    # labels
    label_name = scope.get_unique_variable_name('label_name')
    container.add_node('ArgMax', red_name, label_name,
                       name=scope.get_unique_operator_name('ArgMax'), axis=1)
    _finalize_converter_classes(scope, label_name, operator.outputs[0].full_name, container,
                                op.classes_)


register_converter('SklearnVotingClassifier', convert_voting_classifier)
