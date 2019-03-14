# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import warnings
from ..common._registration import register_converter


def convert_sklearn_text_vectorizer(scope, operator, container):
    """
    Converters for class
    `TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_.
    The current implementation is a work in progress and the ONNX version
    does not produce the exact same results. The converter lets the user
    change some of its parameters.

    Additional options
    ------------------

    regex: string
        The default will change to true in version 1.6.0.
        The tokenizer splits into words using this regular
        expression or the regular expression specified by
        *scikit-learn* is the value is an empty string.
        See also note below.
        Default value: None
    sep: list of separators
        These separators are used to split a string into words.
        Options *sep* is ignore if options *regex* is not None.
        Default value: ``[' ', '.', '?', ',', ';', ':', '!']``.

    Example (from :ref:`l-example-tfidfvectorizer`):

    ::

        seps = {TfidfVectorizer: {"sep": [' ', '.', '?', ',', ';', ':', '!', '(', ')',
                                           '\\n', '"', "'", "-", "[", "]", "@"]}}
        model_onnx = convert_sklearn(pipeline, "tfidf",
                                     initial_types=[("input", StringTensorType([1, 2]))],
                                     options=seps)

    The default regular expression of the tokenizer is ``(?u)\\\\b\\\\w\\\\w+\\\\b``
    (see `re <https://docs.python.org/3/library/re.html>`_).
    This expression may not supported by the library handling the backend.
    `onnxruntime <https://github.com/Microsoft/onnxruntime>`_ uses
    `re2 <https://github.com/google/re2>`_. You may need to switch
    to a custom tokenizer based on
    `python wrapper for re2 <https://pypi.org/project/re2/>_`
    or its sources `pyre2 <https://github.com/facebook/pyre2>`_
    (`syntax <https://github.com/google/re2/blob/master/doc/syntax.txt>`_).
    If the regular expression is not specified and if
    the instance of TfidfVectorizer is using the default
    pattern ``(?u)\\\\b\\\\w\\\\w+\\\\b``, it is replaced by
    ``\\\\\\\\b\\\\\\\\w\\\\\\\\w+\\\\\\\\b``. Any other case has to be
    manually handled.

    Regular expression ``[^\\\\\\\\n]`` is used to split
    a sentance into character (and not works) if ``analyser=='char'``.
    The mode ``analyser=='char_wb'`` is not implemented.
    ````
    
    """ # noqa

    op = operator.raw_operator

    if op.analyzer == "char_wb":
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only tokenizer='word' is supported.")
    if op.strip_accents is not None:
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only stip_accents=None is supported.")

    options = container.get_options(
            op, dict(sep="DEFAULT",
                     regex=None))
    if set(options) != {'sep', 'regex'}:
        raise RuntimeError("Unknown option {} for {}".format(
                                set(options) - {'sep'}, type(op)))

    if op.analyzer == 'word':
        default_pattern = '(?u)\\b\\w\\w+\\b'
        if options['sep'] == "DEFAULT" and options['regex'] is None:
            warnings.warn("Converter for TfidfVectorizer will use scikit-learn regular expression by default in version 1.6.",
                          DeprecationWarning)
            default_separators = [' ', '.', '?', ',', ';', ':', '!']
            regex = op.token_pattern
            if regex == default_pattern:
                regex = '(?U)\\b\\w\\w+\\b'
            default_separators = None
        elif options['regex'] is not None:
            if options['regex']:
                regex = options['regex']
            else:
                regex = op.token_pattern
                if regex == default_pattern:
                    regex = '(?U)\\b\\w\\w+\\b'
            default_separators = None
        else:
            regex = None
            default_separators = options['sep']
    else:
        if options['sep'] != 'DEFAULT':
            raise RuntimeError("Option sep has not effect if analyser != 'word'.")
        if options['regex']:
            regex = options['regex']
        else:
            regex = '.'
        default_separators = None

    if op.preprocessor is not None:
        raise NotImplementedError(
            "Custom preprocessor cannot be converted into ONNX.")
    if op.tokenizer is not None:
        raise NotImplementedError(
            "Custom tokenizer cannot be converted into ONNX.")
    if op.strip_accents is not None:
        raise NotImplementedError(
            "Operator StringNormalizer cannot remove accents.")

    if op.lowercase or op.stop_words_:
        # StringNormalizer

        op_type = 'StringNormalizer'
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        attrs.update({
            'case_change_action': 'LOWER',
            'is_case_sensitive': not op.lowercase,
        })
        if op.stop_words_:
            attrs['stopwords'] = list(sorted(op.stop_words_))
        normalized = scope.get_unique_variable_name('normalized')
        container.add_node(op_type, operator.input_full_names,
                           normalized, op_version=10,
                           op_domain='ai.onnx', **attrs)
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
        'mark': False,
        'mincharnum': 1,
    })
    if regex is None:
        attrs['separators'] = default_separators
    else:
        attrs['tokenexp'] = regex

    tokenized = scope.get_unique_variable_name('tokenized')
    container.add_node(op_type, normalized, tokenized,
                       op_domain='com.microsoft', **attrs)

    # Flatten
    # Tokenizer outputs shape {1, C} or {1, 1, C}.
    # Second shape is not allowed by TfIdfVectorizer.
    # We use Flatten which produces {1, C} in both cases.
    flatt_tokenized = scope.get_unique_variable_name('flattened')
    container.add_node("Flatten", tokenized, flatt_tokenized,
                       name=scope.get_unique_operator_name('Flatten'))
    tokenized = flatt_tokenized

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
    attrs = {'name': scope.get_unique_operator_name("TfIdfVectorizer")}
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
