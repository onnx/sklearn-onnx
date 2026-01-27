"""
Implement a custom converter with ndonnx
========================================

`ndonnx <https://ndonnx.readthedocs.io/>`_ aims at converting a code using numpy
into ONNX. It is tracing the execution path by storing every operation
applied on a tracing object. It may work with a scikit-learn if it supports
the Array API. That means every estimator in a pipeline must be part of the
following list `Array API support (Estimators)
<https://scikit-learn.org/stable/modules/array_api.html#estimators>`_.

Nevertheless can still be used to smoothly convert an estimator written with
numpy.


Estimator
+++++++++

A very simple pipeline and the first attempt to convert a custom estimator into ONNX.
"""

import numpy as np
from numpy.testing import assert_allclose
import ndonnx as ndx
from onnx.version_converter import convert_version
from pandas import DataFrame
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx import to_onnx

# For the custom converter
from skl2onnx import update_registered_converter
from skl2onnx.common.utils import check_input_and_output_numbers
from skl2onnx.helpers import add_onnx_graph

# To check discrepancies
from onnxruntime import InferenceSession


class GrowthCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_growth(self, x, y):
        return (x - y) / y * 100

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.apply(lambda x: self.calculate_growth(x.a, x.b), axis=1)
        return x.values.reshape((-1, 1))


mapper = ColumnTransformer(
    transformers=[
        ("ab", FunctionTransformer(), ["a", "b"]),  # We keep the first column.
        ("c", GrowthCalculator(), ["a", "b"]),  # We add a new one.
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

data = DataFrame(
    [
        dict(a=2, b=1, f=5),
        dict(a=50, b=4, f=10),
        dict(a=5, b=2, f=4),
        dict(a=100, b=6, f=20),
    ]
).astype(np.float32)
y = np.array([0, 1, 0, 1], dtype=np.int64)

pipe_tr = Pipeline([("mapper", mapper), ("classifier", DecisionTreeClassifier())])
pipe_tr.fit(data, y)

# %%
# The conversion fails with an error message.

try:
    to_onnx(pipe_tr, data[:1], options={"zipmap": False})
except Exception as e:
    print("It does not work:", e)


# %%
# Custom converter with ndonnx
# ++++++++++++++++++++++++++++++++
#
# ONNX is more precise than numpy about shapes so it is important
# to make sure the shapes of the inputs and outputs are 2D.
# Inputs of the converter are creared with ``ndx.argument``.
# Constants are not just python constant but typed arrays
# to avoid missing casts in ONNX.


def growth_converter_ndonnx(scope, operator, container):
    # No need to retrieve the fitted estimator, it is not trained.
    # op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(operator.inputs[0].type)

    # use of ndonnx to write the model,
    # we create dummy inputs
    x = ndx.argument(
        shape=("N", 2), dtype=ndx.float64 if dtype == np.float64 else ndx.float32
    )

    # the expression to convert, it could be move into a function
    # called both by the converter and the model
    # (X[0] - X[1]) / X[1] * 100
    growth = (x[:, :1] - x[:, :1]) / x[:, :1] * ndx.asarray([100], dtype=x.dtype)

    # conversion into onnx, how to specify the opset?
    proto = ndx.build({"x": x}, {"y": growth})

    # The function is written with opset 18, it needs to be converted
    # to the opset required by the user when the conversion starts.
    proto_version = convert_version(proto, opv)
    add_onnx_graph(scope, operator, container, proto_version)


def growth_shape_calculator(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Gets the input type, the transformer works on any numerical type.
    input_type = operator.inputs[0].type.__class__
    # The first dimension is usually dynamic (batch dimension).
    input_dim = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = input_type([input_dim, 1])


update_registered_converter(
    GrowthCalculator,
    "AliasGrowthCalculator",
    growth_shape_calculator,
    growth_converter_ndonnx,
)

# %%
# Let's check it works.

onx = to_onnx(pipe_tr, data[:1], target_opset=22, options={"zipmap": False})
import onnx

onnx.save(onx, "h.onnx")


# %%
# And again the discrepancies.

expected = (pipe_tr.predict(data), pipe_tr.predict_proba(data))
feeds = {"a": data[["a"]].values, "b": data[["b"]].values, "f": data[["f"]].values}

ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
got = ref.run(None, feeds)
assert_allclose(expected[1], got[1])
assert_allclose(expected[0], got[0])


# %%
# Finally.
print("done.")
