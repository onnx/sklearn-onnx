# SPDX-License-Identifier: Apache-2.0

"""
TfIdf, SVC and sparse matrices
==============================

.. index:: sparse

The example is useful to whom wants to convert a pipeline
doing a TfIdfVectorizer + SVC when the features are sparse.

The pipeline
++++++++++++
"""
import os
import pickle
from typing import Any
import numpy as np
from numpy import ndarray
import scipy
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from onnxruntime import InferenceSession
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.data_types import StringTensorType

X_train = np.array(
    [
        "This is the first document",
        "This document is the second document.",
        "And this is the third one",
        "Is this the first document?",
    ]
).reshape((4, 1))
y_train = np.array([0, 1, 0, 1])

model_pipeline = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                use_idf=True,
                ngram_range=(1, 3),
                max_features=30000,
            ),
        ),
        (
            "classifier",
            SVC(
                class_weight="balanced",
                kernel="rbf",
                gamma="scale",
                probability=True,
            ),
        ),
    ]
)
model_pipeline.fit(X_train.ravel(), y_train)

out0 = model_pipeline.steps[0][-1].transform(X_train.ravel())
is_sparse = isinstance(out0, scipy.sparse._csr.csr_matrix)
print(f"Output type for TfIdfVectorizier is {'sparse' if is_sparse else 'dense'}.")

svc_coef = model_pipeline.steps[1][-1].support_vectors_
is_parse = isinstance(svc_coef, scipy.sparse._csr.csr_matrix)
print(f"Supports for SVC is {'sparse' if is_sparse else 'dense'}.")
sparsity = 1 - (svc_coef != 0).sum() / np.prod(svc_coef.shape)
print(f"sparsity={sparsity} and shape={svc_coef.shape}")


######################################
# Size Comparison
# +++++++++++++++

pkl_name = "model.pkl"
with open(pkl_name, "wb") as f:
    pickle.dump(model_pipeline, f)

onx_name = "model.onnx"
onx = to_onnx(
    model_pipeline,
    initial_types=[("input", StringTensorType([None, 1]))],
    options={SVC: {"zipmap": False}},
    target_opset=18,
)
with open(onx_name, "wb") as f:
    f.write(onx.SerializeToString())

print(f"pickle size={os.stat(pkl_name).st_size}")
print(f"onnx size={os.stat(onx_name).st_size}")

#######################################
# On such small model, it does not show that SVC is using a sparse matrix
# and ONNX SVMClassifier is using a dense one. If the matrix is 90% sparse,
# this part becomes 10 times bigger once converter into ONNX.
#
# Tweak
# +++++
#
# The idea is to take out the matrix of coefficient out of SVC by
# reducing the number dimensions.
# We could apply a PCA but it does not support sparse features.
# TruncatedSVD does but the matrix it produces to reduce the dimension
# is dense. SparsePCA does not support sparse feature as well.
# Let's try something custom: a TruncatedSVD and then some small coefficient
# will be set to zero.


class SparseTruncatedSVD(TruncatedSVD):
    def __init__(
        self,
        n_components=2,
        *,
        algorithm="randomized",
        n_iter=5,
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
        tol=0.0,
        sparsity=0.9,
    ):
        TruncatedSVD.__init__(
            self,
            n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
            tol=tol,
        )
        self.sparsity = sparsity

    def fit_transform(self, X, y=None):
        TruncatedSVD.fit_transform(self, X, y)

        # The matrix. We could choose the coefficients to set to zero
        # by minimizing `(X @ M.T - X @ M0.T) ** 2`
        # where M is the original matrix and M0 the new one.
        # In a first approach, we just sort the coefficients by absolute value.
        components = self.components_.ravel()
        flat = list((v, i) for i, v in enumerate(np.abs(components)))
        flat.sort()
        last_index = int(self.sparsity * len(flat))
        for tu in flat[:last_index]:
            components[tu[1]] = 0
        self.components_ = scipy.sparse.coo_matrix(
            components.reshape(self.components_.shape)
        )
        return self.transform(X)


sparse_pipeline = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                use_idf=True,
                ngram_range=(1, 3),
                max_features=30000,
            ),
        ),
        ("sparse", SparseTruncatedSVD(10, sparsity=0.6)),
        (
            "classifier",
            SVC(
                class_weight="balanced",
                kernel="rbf",
                gamma="scale",
                probability=True,
            ),
        ),
    ]
)
sparse_pipeline.fit(X_train.ravel(), y_train)

expected = model_pipeline.predict(X_train.ravel())
got = sparse_pipeline.predict(X_train.ravel())
print(f"Number of different predicted labels: {((expected-got)==0).sum()}")

expected = model_pipeline.predict_proba(X_train.ravel())
got = sparse_pipeline.predict_proba(X_train.ravel())
diff = np.abs(expected - got)
print(f"Average absolute difference for the probabilities: {diff.max(axis=1)}")

######################################
# Conversion to ONNX
# ++++++++++++++++++
#
# The new transformer cannot be converted because sklearn-onnx does not have any
# registered converter for it. We must implement it.
# We use the converter for TruncatedSVD as a base and a sparse matrix multiplication
# implemented in onnxruntime (see `OperatorKernels.md
# <https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md>`_).

from skl2onnx.common._topology import Scope, Operator
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    guess_proto_type,
)


def calculate_sparse_sklearn_truncated_svd_output_shapes(operator):
    cls_type = operator.inputs[0].type.__class__
    if cls_type != DoubleTensorType:
        cls_type = FloatTensorType
    N = operator.inputs[0].get_first_dimension()
    K = operator.raw_operator.n_components
    operator.outputs[0].type = cls_type([N, K])


def convert_sparse_truncated_svd(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    # Create alias for the scikit-learn truncated SVD model we
    # are going to convert
    svd = operator.raw_operator
    if isinstance(operator.inputs[0].type, DoubleTensorType):
        proto_dtype = guess_proto_type(operator.inputs[0].type)
    else:
        proto_dtype = guess_proto_type(FloatTensorType())
    # Transpose [K, C] matrix to [C, K], where C/K is the
    # input/transformed feature dimension
    transform_matrix = svd.components_
    transform_matrix_name = scope.get_unique_variable_name("transform_matrix")
    # Put the transformation into an ONNX tensor
    container.add_initializer(
        transform_matrix_name,
        proto_dtype,
        transform_matrix.shape,
        transform_matrix,
    )

    input_name = operator.inputs[0].full_name

    transposed_inputs = scope.get_unique_variable_name("transposed_inputs")
    container.add_node("Transpose", input_name, transposed_inputs, perm=[1, 0])

    transposed_outputs = scope.get_unique_variable_name("transposed_outputs")
    container.add_node(
        "SparseToDenseMatMul",
        [transform_matrix_name, transposed_inputs],
        transposed_outputs,
        op_domain="com.microsoft",
        op_version=1,
    )
    container.add_node(
        "Transpose", transposed_outputs, operator.outputs[0].full_name, perm=[1, 0]
    )


update_registered_converter(
    SparseTruncatedSVD,
    "SparseTruncatedSVD",
    calculate_sparse_sklearn_truncated_svd_output_shapes,
    convert_sparse_truncated_svd,
)

sparse_onx_name = "model_sparse.onnx"
sparse_onx = to_onnx(
    sparse_pipeline,
    initial_types=[("input", StringTensorType([None, 1]))],
    options={SVC: {"zipmap": False}},
    target_opset=18,
)
print(sparse_onx)
with open(sparse_onx_name, "wb") as f:
    f.write(sparse_onx.SerializeToString())

print(f"pickle size={os.stat(pkl_name).st_size}")
print(f"onnx size={os.stat(onx_name).st_size}")
print(f"sparse onnx size={os.stat(sparse_onx_name).st_size}")

############################################
# Let's check it is working with onnxruntime.

sess = InferenceSession(sparse_onx_name, providers=["CPUExecutionProvider"])
got = sess.run(None, {"input": X_train})
print(got)


######################################
# Conclusion
# ++++++++++
#
# This option decreases the size of the onnx model by using one
# sparse matrix in the converted pipeline. It may bring an accuracy loss.
