# SPDX-License-Identifier: Apache-2.0

"""
FeatureHasher, pandas values and unexpected discrepancies
=========================================================

A game of finding it goes wrong and there are multiple places.


Initial example
+++++++++++++++
"""
import numpy as np
from pandas import DataFrame
from onnxruntime import InferenceSession
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

df = DataFrame(
    {
        "Cat1": ["a", "b", "d", "abd", "e", "z", "ez"],
        "Cat2": ["A", "B", "D", "ABD", "e", "z", "ez"],
        "Label": [1, 1, 0, 0, 1, 0, 0],
    }
)

cat_features = [c for c in df.columns if "Cat" in c]
X_train = df[cat_features]

X_train["cat_features"] = df[cat_features].values.tolist()
X_train = X_train.drop(cat_features, axis=1)
y_train = df["Label"]

pipe = Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                [
                    (
                        "cat_preprocessor",
                        FeatureHasher(
                            n_features=8,
                            input_type="string",
                            alternate_sign=False,
                            dtype=np.float32,
                        ),
                        "cat_features",
                    )
                ],
                sparse_threshold=0.0,
            ),
        ),
        ("classifier", GradientBoostingClassifier(n_estimators=2, max_depth=2)),
    ],
)
pipe.fit(X_train, y_train)


###################################
# Conversion to ONNX.

onx = to_onnx(
    pipe,
    initial_types=[("cat_features", StringTensorType([None, None]))],
    options={"zipmap": False},
)

###################################
# There are many discrepancies?

expected_proba = pipe.predict_proba(X_train)
sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])


got = sess.run(None, dict(cat_features=X_train.values))


print("expected")
print(expected_proba)

print("onnx")
print(got[1])

#########################################
# Let's check the feature hasher
# ++++++++++++++++++++++++++++++
#
# We just remove the classifier.

pipe_hash = Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                [
                    (
                        "cat_preprocessor",
                        FeatureHasher(
                            n_features=8,
                            input_type="string",
                            alternate_sign=False,
                            dtype=np.float32,
                        ),
                        "cat_features",
                    )
                ],
                sparse_threshold=0.0,
            ),
        ),
    ],
)
pipe_hash.fit(X_train, y_train)

onx = to_onnx(
    pipe_hash,
    initial_types=[("cat_features", StringTensorType([None, None]))],
    options={"zipmap": False},
)

expected = pipe_hash.transform(X_train)
sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])


got = sess.run(None, dict(cat_features=X_train.values))


print("expected")
print(expected)

print("onnx")
print(got[0])


#############################################
# First Error
# +++++++++++
#
# The instruction
# ``X_train["cat_features"] = df[cat_features].values.tolist()``
# creates a DataFrame with on column of a lists of two values.
# The type list is expected by scikit-learn and it can process a variable
# number of elements per list. onnxruntime cannot do that.
# It must be changed into the following.

pipe_hash = Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                [
                    (
                        "cat_preprocessor1",
                        FeatureHasher(
                            n_features=8,
                            input_type="string",
                            alternate_sign=False,
                            dtype=np.float32,
                        ),
                        [0],
                    ),
                    (
                        "cat_preprocessor2",
                        FeatureHasher(
                            n_features=8,
                            input_type="string",
                            alternate_sign=False,
                            dtype=np.float32,
                        ),
                        [1],
                    ),
                ],
                sparse_threshold=0.0,
            ),
        ),
    ],
)

X_train_skl = df[cat_features].copy()
for c in cat_features:
    X_train_skl[c] = X_train_skl[c].values.tolist()

pipe_hash.fit(X_train_skl.values, y_train)

onx = to_onnx(
    pipe_hash,
    initial_types=[
        ("cat1", StringTensorType([None, 1])),
        ("cat2", StringTensorType([None, 1])),
    ],
    options={"zipmap": False},
)


expected = pipe_hash.transform(X_train_skl.values)
sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])


got = sess.run(
    None,
    dict(
        cat1=df["Cat1"].values.reshape((-1, 1)), cat2=df["Cat2"].values.reshape((-1, 1))
    ),
)


print("expected")
print(expected)

print("onnx")
print(got[0])

###########################################
# This is not the original pipeline. It has 16 columns instead of 8
# but it does produce the same results.
# One option would be to add the first 8 columns to the other 8
# by using a custom converter.

# to be continued...

############################################
# Sparsity?
# +++++++++
#
# Let's try with the classifier now and no `sparse_threshold=0.0`.

pipe = Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                [
                    (
                        "cat_preprocessor1",
                        FeatureHasher(
                            n_features=8,
                            input_type="string",
                            alternate_sign=False,
                            dtype=np.float32,
                        ),
                        [0],
                    ),
                    (
                        "cat_preprocessor2",
                        FeatureHasher(
                            n_features=8,
                            input_type="string",
                            alternate_sign=False,
                            dtype=np.float32,
                        ),
                        [1],
                    ),
                ],
                # sparse_threshold=0.0,
            ),
        ),
        ("classifier", GradientBoostingClassifier(n_estimators=2, max_depth=2)),
    ],
)

X_train_skl = df[cat_features].copy()
for c in cat_features:
    X_train_skl[c] = X_train_skl[c].values.tolist()

pipe.fit(X_train_skl.values, y_train)

onx = to_onnx(
    pipe,
    initial_types=[
        ("cat1", StringTensorType([None, 1])),
        ("cat2", StringTensorType([None, 1])),
    ],
    options={"zipmap": False},
)


expected = pipe.predict_proba(X_train_skl.values)
sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])


got = sess.run(
    None,
    dict(
        cat1=df["Cat1"].values.reshape((-1, 1)), cat2=df["Cat2"].values.reshape((-1, 1))
    ),
)


print("expected")
print(expected)

print("onnx")
print(got[1])

###########################################
# scikit-learn keeps the sparse outputs from
# the FeatureHasher. onnxruntime does not support
# dense feautres. This may have an impact on the conversion
# if the model next to this steps makes a different between a
# missing sparse value and zero.

print(pipe.steps[0][-1].transform(X_train_skl.values))
