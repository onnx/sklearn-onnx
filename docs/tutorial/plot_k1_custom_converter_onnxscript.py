"""
Implement a custom converter with onnxscript
============================================

A pipeline including a custom estimator
cannot be automatically converted into onnx because there is no converter able to
convert custom python code into ONNX. A custom converter needs to be written
specifically for it.

Estimator
+++++++++

A very simple pipeline and the first attempt to convert a custom estimator into ONNX.
"""

import numpy as np
from numpy.testing import assert_allclose
from onnx.version_converter import convert_version
from pandas import DataFrame
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx

# For the custom converter
from skl2onnx import update_registered_converter
from skl2onnx.common.utils import check_input_and_output_numbers
from skl2onnx.helpers import add_onnx_graph
import onnxscript
from onnxscript import opset18 as op

# To check discrepancies
from onnxruntime import InferenceSession


class GrowthCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_growth(self, x, y):
        return 100 * (x - y) / y

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
)
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
# Custom converter with onnxscript
# ++++++++++++++++++++++++++++++++
#
# `onnxscript <https://github.com/microsoft/onnxscript>`_
# offers a less verbose API than what onnx package implements.
# Let's see how to use it to write the converters.


@onnxscript.script()
def calculate_onnxscript(X):
    # onnxscript must define an opset. We use an identity node
    # from a specific opset to set it (otherwise it fails).
    xi = op.Identity(X)
    x0 = xi[:, :1]
    x1 = xi[:, 1:]
    return (x0 - x1) / x1 * 100


# %%
# Let's use it in the converter.


def growth_converter_onnxscript(scope, operator, container):
    # No need to retrieve the fitted estimator, it is not trained.
    # op = operator.raw_operator
    opv = container.target_opset

    # 100 * (x-y)/y  --> 100 * (X[0] - X[1]) / X[1]
    proto = calculate_onnxscript.to_model_proto()
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
    growth_converter_onnxscript,
)

# %%
# Let's check it works.

onx = to_onnx(pipe_tr, data[:1], target_opset=18, options={"zipmap": False})


# %%
# And again the discrepancies.

expected = (pipe_tr.predict(data), pipe_tr.predict_proba(data))
feeds = {"a": data[["a"]].values, "b": data[["b"]].values, "f": data[["f"]].values}

ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
got = ref.run(None, feeds)
assert_allclose(expected[0], got[0])
assert_allclose(expected[1], got[1])


# %%
# Finally.
print("done.")
