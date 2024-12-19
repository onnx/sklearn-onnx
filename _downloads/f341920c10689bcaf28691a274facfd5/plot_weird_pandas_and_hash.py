# SPDX-License-Identifier: Apache-2.0

"""
FeatureHasher, pandas values and unexpected discrepancies
=========================================================

A game of finding it goes wrong and there are multiple places.


Initial example
+++++++++++++++
"""

import logging
import numpy as np
from pandas import DataFrame
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

log = logging.getLogger("skl2onnx")
log.setLevel(logging.ERROR)


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


print("expected probabilities")
print(expected_proba)

print("onnx probabilities")
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


print("expected hashed features")
print(expected)

print("onnx hashed features")
print(got[0])

#######################################
# Nothing seems to be working.
#
# First proposal
# ++++++++++++++
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


print("expected fixed hashed features")
print(expected)

print("onnx fixed hashed features")
print(got[0])

###########################################
# This is not the original pipeline. It has 16 columns instead of 8
# but it does produce the same results.
# One option would be to add the first 8 columns to the other 8
# by using a custom converter.
#
# Second proposal
# +++++++++++++++
#
# We use the same initial pipeline but we tweak the input
# onnxruntime receives.

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
    initial_types=[("cat_features", StringTensorType([None, 1]))],
    options={"zipmap": False, "preprocessor__cat_preprocessor__separator": "#"},
)

expected = pipe_hash.transform(X_train)


so = SessionOptions()
so.register_custom_ops_library(get_library_path())
sess = InferenceSession(onx.SerializeToString(), so, providers=["CPUExecutionProvider"])

# We merged both columns cat1 and cat2 into a single cat_features.
df_fixed = DataFrame()
df_fixed["cat_features"] = np.array([f"{a}#{b}" for a, b in X_train["cat_features"]])

got = sess.run(None, {"cat_features": df_fixed[["cat_features"]].values})

print("expected original hashed features")
print(expected)

print("onnx fixed original hashed features")
print(got[0])

############################################
# It works now.
#
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
                # sparse_threshold=0.0,
            ),
        ),
        ("classifier", GradientBoostingClassifier(n_estimators=2, max_depth=2)),
    ],
)
pipe.fit(X_train, y_train)
expected = pipe.predict_proba(X_train)


onx = to_onnx(
    pipe,
    initial_types=[("cat_features", StringTensorType([None, 1]))],
    options={"zipmap": False, "preprocessor__cat_preprocessor__separator": "#"},
)

so = SessionOptions()
so.register_custom_ops_library(get_library_path())
sess = InferenceSession(onx.SerializeToString(), so, providers=["CPUExecutionProvider"])
got = sess.run(None, {"cat_features": df_fixed[["cat_features"]].values})


print("expected probabilies")
print(expected)

print("onnx probabilies")
print(got[1])

###########################################
# scikit-learn keeps the sparse outputs from
# the FeatureHasher. onnxruntime does not support
# sparse features. This may have an impact on the conversion
# if the model next to this step makes a difference between a
# missing sparse value and zero.
# That does not seem to be the case for this model but
# other models or libraries may behave differently.

print(pipe.steps[0][-1].transform(X_train))
