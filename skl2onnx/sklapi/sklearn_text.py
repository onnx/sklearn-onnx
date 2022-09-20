"""
@file
@brief Overloads :epkg:`TfidfVectorizer` and :epkg:`CountVectorizer`.
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
try:
    from sklearn.feature_extraction.text import (
        _VectorizerMixin as VectorizerMixin)
except ImportError:  # pragma: no cover
    # scikit-learn < 0.23
    from sklearn.feature_extraction.text import VectorizerMixin


class NGramsMixin(VectorizerMixin):
    """
    Overloads method `_word_ngrams
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L148>`_
    to get tuples instead of string in member `vocabulary_
    <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_.
    of `TfidfVectorizer` or :epkg:`CountVectorizer`.
    It contains the list of n-grams used to process documents.
    See :class:`TraceableCountVectorizer` and :class:`TraceableTfidfVectorizer`
    for example.
    """

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if tokens is not None:
            new_tokens = []
            for token in tokens:
                new_tokens.append(
                    (token,) if isinstance(token, str) else token)
            tokens = new_tokens

        if stop_words is not None:
            tokens = [(w, ) for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append

            def space_join(tokens):
                new_tokens = []
                for token in tokens:
                    if isinstance(token, str):
                        new_tokens.append(token)
                    elif isinstance(token, tuple):
                        new_tokens.extend(token)
                    else:
                        raise TypeError(  # pragma: no cover
                            f"Unable to build a n-grams out of {tokens}.")
                return tuple(new_tokens)

            for n in range(min_n,
                           min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))
        return tokens


class TraceableCountVectorizer(CountVectorizer, NGramsMixin):
    """
    Inherits from :class:`NGramsMixin` which overloads method `_word_ngrams
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L148>`_
    to keep more information about n-grams but still produces the same
    outputs than `CountVectorizer`.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.feature_extraction.text import CountVectorizer
        from mlinsights.mlmodel.sklearn_text import TraceableCountVectorizer
        from pprint import pformat

        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "Is this the first document?",
            "",
        ]).reshape((4, ))

        print('CountVectorizer from scikit-learn')
        mod1 = CountVectorizer(ngram_range=(1, 2))
        mod1.fit(corpus)
        print(mod1.transform(corpus).todense()[:2])
        print(pformat(mod1.vocabulary_)[:100])

        print('TraceableCountVectorizer from scikit-learn')
        mod2 = TraceableCountVectorizer(ngram_range=(1, 2))
        mod2.fit(corpus)
        print(mod2.transform(corpus).todense()[:2])
        print(pformat(mod2.vocabulary_)[:100])

    A weirder example with
    @see cl TraceableTfidfVectorizer shows more differences.
    """

    def _word_ngrams(self, tokens, stop_words=None):
        return NGramsMixin._word_ngrams(
            self, tokens=tokens, stop_words=stop_words)


class TraceableTfidfVectorizer(TfidfVectorizer, NGramsMixin):
    """
    Inherits from :class:`NGramsMixin` which overloads method `_word_ngrams
    <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L148>`_
    to keep more information about n-grams but still produces the same
    outputs than `TfidfVectorizer`.

    .. runpython::
        :showcode:

        import numpy
        from sklearn.feature_extraction.text import TfidfVectorizer
        from mlinsights.mlmodel.sklearn_text import TraceableTfidfVectorizer
        from pprint import pformat

        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "Is this the first document?",
            "",
        ]).reshape((4, ))

        print('TfidfVectorizer from scikit-learn')
        mod1 = TfidfVectorizer(ngram_range=(1, 2),
                               token_pattern="[a-zA-Z ]{1,4}")
        mod1.fit(corpus)
        print(mod1.transform(corpus).todense()[:2])
        print(pformat(mod1.vocabulary_)[:100])

        print('TraceableTfidfVectorizer from scikit-learn')
        mod2 = TraceableTfidfVectorizer(ngram_range=(1, 2),
                                       token_pattern="[a-zA-Z ]{1,4}")
        mod2.fit(corpus)
        print(mod2.transform(corpus).todense()[:2])
        print(pformat(mod2.vocabulary_)[:100])
    """

    def _word_ngrams(self, tokens, stop_words=None):
        return NGramsMixin._word_ngrams(
            self, tokens=tokens, stop_words=stop_words)
