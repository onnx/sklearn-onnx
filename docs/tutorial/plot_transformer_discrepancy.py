"""
.. _example-transform-discrepancy:

Dealing with discrepancies (tf-idf)
===================================

.. index:: td-idf

`TfidfVectorizer <https://scikit-learn.org/stable/modules/
generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_
is one transform for which the corresponding converted onnx model
may produce different results. The larger the vocabulary is,
the higher the probability to get different result is.
This example proposes a equivalent model with no discrepancies.

Imports, setups
+++++++++++++++

All imports. It also registered onnx converters for :epkg:`xgboost`
and :epkg:`lightgbm`.
"""

import pprint
import numpy
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from onnxruntime import InferenceSession
from skl2onnx import to_onnx


def print_sparse_matrix(m):
    nonan = numpy.nan_to_num(m)
    mi, ma = nonan.min(), nonan.max()
    if mi == ma:
        ma += 1
    mat = numpy.empty(m.shape, dtype=numpy.str_)
    mat[:, :] = "."
    if hasattr(m, "todense"):
        dense = m.todense()
    else:
        dense = m
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if dense[i, j] > 0:
                c = int((dense[i, j] - mi) / (ma - mi) * 25)
                mat[i, j] = chr(ord("A") + c)
    return "\n".join("".join(line) for line in mat)


def diff(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Cannot compare matrices with different shapes {a.shape} != {b.shape}."
        )
    d = numpy.abs(a - b).sum() / a.size
    return d


##########################################
# Artificial datasets
# +++++++++++++++++++
#
# Iris + a text column.


strings = numpy.array(
    [
        "This a sentence.",
        "This a sentence with more characters $^*&'(-...",
        """var = ClassName(var2, user=mail@anywhere.com, pwd"""
        """=")_~-('&]@^\\`|[{#")""",
        "c79857654",
        "https://complex-url.com/;76543u3456?g=hhh&amp;h=23",
        "01-03-05T11:12:13",
        "https://complex-url.com/;dd76543u3456?g=ddhhh&amp;h=23",
    ]
).reshape((-1, 1))

pprint.pprint(strings)

############################################
# Fit a TfIdfVectorizer
# +++++++++++++++++++++

tfidf = Pipeline([("pre", ColumnTransformer([("tfidf", TfidfVectorizer(), 0)]))])

#############################
# We leave a couple of strings out of the training set.

tfidf.fit(strings[:-2])
tr = tfidf.transform(strings)
tfidf_step = tfidf.steps[0][1].transformers_[0][1]
# print(f"output columns: {tfidf_step.get_feature_names_out()}")
print("rendered outputs")
print(print_sparse_matrix(tr))

#############################################
# Conversion to ONNX
# ++++++++++++++++++

onx = to_onnx(tfidf, strings)


############################################
# Execution with ONNX
# +++++++++++++++++++

sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
got = sess.run(None, {"X": strings})[0]
print(f"differences={diff(tr, got):g}")
print(print_sparse_matrix(got))
