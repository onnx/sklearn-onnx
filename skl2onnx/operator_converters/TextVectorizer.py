# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import warnings
from ..common._registration import register_converter


def convert_sklearn_text_vectorizer(scope, operator, container):

    op = operator.raw_operator

    if op.analyzer != "word":
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only tokenizer='word' is supported.")
    if op.strip_accents is not None:
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only stip_accents=None is supported.")

    default_pattern = '(?u)\\b\\w\\w+\\b'
    default_separators = [' ', '.', '?', ',', ';', ':', '!']
    if op.token_pattern != default_pattern:
        raise NotImplementedError(
            "Only the default tokenizer based on default regular expression "
            "'{0}' is implemented.".format(default_pattern))
    if op.preprocessor is not None:
        raise NotImplementedError(
            "Custom preprocessor cannot be converted into ONNX.")
    if op.tokenizer is not None:
        raise NotImplementedError(
            "Custom tokenizer cannot be converted into ONNX.")
    if op.strip_accents is not None:
        raise NotImplementedError(
            "Operator StringNormalizer cannot remove accents.")

    msg = ("The default regular expression '{0}' splits strings based on "
           "anything but a space. The current specification splits strings "
           "based on the following separators {1}.")
    warnings.warn(msg.format(default_pattern, default_separators))

    if op.lowercase or op.stop_words_:
        # StringNormalizer

        op_type = 'StringNormalizer'
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        attrs.update({
            'casechangeaction': 'LOWER',
            'is_case_sensitive': not op.lowercase,
        })
        if op.stop_words_:
            attrs['stopwords'] = list(sorted(op.stop_words_))
        normalized = scope.get_unique_variable_name('normalized')
        container.add_node(op_type, operator.input_full_names,
                           normalized, op_domain='com.microsoft', **attrs)
    else:
        normalized = operator.input_full_names

    # Tokenizer
    padvalue = "#"
    while padvalue in op.vocabulary_:
        padvalue += "#"

    op_type = 'Tokenizer'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs.update({
        'pad_value': padvalue,
        'separators': default_separators,
        'mark': False,
        'mincharnum': 1,
    })

    tokenized = scope.get_unique_variable_name('tokenized')
    container.add_node(op_type, normalized, tokenized,
                       op_domain='com.microsoft', **attrs)

    # Ngram - TfIdfVectorizer
    C = max(op.vocabulary_.values()) + 1
    words = [None for i in range(C)]
    weights = [0 for i in range(C)]
    if hasattr(op, "idf_"):
        for k, v in op.vocabulary_.items():
            words[v] = k
            weights[v] = op.idf_[v]
        mode = 'TFIDF'
    else:
        for k, v in op.vocabulary_.items():
            words[v] = k
            weights[v] = 1.
        mode = 'IDF' if hasattr(op, 'use_idf') else 'TF'

    # Scikit-learn sorts n-grams by alphabetical order..
    # onnx assumes it is sorted by n.
    split_words = [(w.split(), w) for w in words]
    ng_split_words = [(len(a[0]), a[0], i) for i, a in enumerate(split_words)]
    ng_split_words.sort()
    key_indices = [a[2] for a in ng_split_words]
    ngcounts = [0 for i in range(op.ngram_range[0])]

    words = list(ng_split_words[0][1])
    for i in range(1, len(ng_split_words)):
        if ng_split_words[i-1][0] != ng_split_words[i][0]:
            ngcounts.append(len(words))
        words.extend(ng_split_words[i][1])

    weights_ = [weights[a[2]] for a in ng_split_words]
    weights = list(weights_)
    for i, ind in enumerate(key_indices):
        weights[ind] = weights_[i]

    # Create the node.
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs.update({
        'min_gram_length': op.ngram_range[0],
        'max_gram_length': op.ngram_range[1],
        'mode': mode,
        'max_skip_count': 0,
        'pool_strings': words,
        'ngram_indexes': key_indices,
        'ngram_counts': ngcounts,
        'weights': weights,
    })

    if getattr(op, 'norm', None) is None:
        output = operator.output_full_names
    else:
        notnormalized = scope.get_unique_variable_name('notnormalized')
        output = [notnormalized]

    if container.target_opset < 9:
        op_type = 'Ngram'
        container.add_node(op_type, tokenized, output,
                           op_domain='com.microsoft', **attrs)
    else:
        op_type = 'TfIdfVectorizer'
        container.add_node(op_type, tokenized, output, op_domain='ai.onnx',
                           op_version=9, **attrs)

    if getattr(op, 'norm', None) is not None:
        op_type = 'Normalizer'
        norm_map = {'max': 'MAX', 'l1': 'L1', 'l2': 'L2'}
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        if op.norm in norm_map:
            attrs['norm'] = norm_map[op.norm]
        else:
            raise RuntimeError('Invalid norm: %s' % op.norm)

        container.add_node(op_type, output, operator.output_full_names,
                           op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnCountVectorizer', convert_sklearn_text_vectorizer)
register_converter('SklearnTfidfVectorizer', convert_sklearn_text_vectorizer)
