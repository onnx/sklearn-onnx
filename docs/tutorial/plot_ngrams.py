# SPDX-License-Identifier: Apache-2.0

"""
.. _example-ngrams:

Tricky issue when converting CountVectorizer or TfidfVectorizer
===============================================================

This issue is described at `scikit-learn/issues/13733
<https://github.com/scikit-learn/scikit-learn/issues/13733>`_.
If a CountVectorizer or a TfidfVectorizer produces a token with a space,
skl2onnx cannot know if it a bi-grams or a unigram with a space.

A simple example impossible to convert
++++++++++++++++++++++++++++++++++++++
"""

import pprint
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import to_onnx
from skl2onnx.sklapi import TraceableTfidfVectorizer
import skl2onnx.sklapi.register  # noqa: F401

corpus = numpy.array(
    [
        "This is the first document.",
        "This document is the second document.",
        "Is this the first document?",
        "",
    ]
).reshape((4,))

pattern = r"\b[a-z ]{1,10}\b"
mod1 = TfidfVectorizer(ngram_range=(1, 2), token_pattern=pattern)
mod1.fit(corpus)


######################################
# Unigrams and bi-grams are placed into the following container
# which maps it to its column index.

pprint.pprint(mod1.vocabulary_)


####################################
# Conversion.

try:
    to_onnx(mod1, corpus)
except RuntimeError as e:
    print(e)


#######################################
# TraceableTfidfVectorizer
# ++++++++++++++++++++++++
#
# Class :class:`TraceableTfidfVectorizer` is equivalent to
# :class:`sklearn.feature_extraction.text.TfidfVectorizer`
# but stores the unigrams and bi-grams of the vocabulary with tuple
# instead of concatenating every piece into a string.


mod2 = TraceableTfidfVectorizer(ngram_range=(1, 2), token_pattern=pattern)
mod2.fit(corpus)

pprint.pprint(mod2.vocabulary_)

#######################################
# Let's check it produces the same results.

assert_almost_equal(mod1.transform(corpus).todense(), mod2.transform(corpus).todense())

####################################
# Conversion. Line `import skl2onnx.sklapi.register`
# was added to register the converters associated to these
# new class. By default, only converters for scikit-learn are
# declared.

onx = to_onnx(mod2, corpus)
sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
got = sess.run(None, {"X": corpus})

###################################
# Let's check if there are discrepancies...

assert_almost_equal(mod2.transform(corpus).todense(), got[0])
