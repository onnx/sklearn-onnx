"""
Issues with FunctionTransformer
===============================

A pipeline including a `FunctionTransformer
<https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_
cannot be automatically converted into onnx because there is no converter able to
convert custom python code into ONNX. A custom converter needs to be written
specifically for it.

Initial try
+++++++++++

A very simple pipeline and the first attempt to convert it into ONNX.
"""
import numpy as np
from numpy.testing import assert_allclose
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
from skl2onnx.algebra.onnx_ops import OnnxSlice, OnnxSub, OnnxDiv, OnnxMul, OnnxCastLike

# To check discrepancies
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession


def calculate(df):
    df["c"] = 100 * (df["a"] - df["b"]) / df["b"]
    return df


mapper = ColumnTransformer(
    transformers=[
        ("c", FunctionTransformer(calculate), ["a", "b"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
mapper.set_output(transform="pandas")

pipe = Pipeline([("mapper", mapper), ("classifier", DecisionTreeClassifier())])

data = DataFrame(
    [
        dict(a=1, b=2, f=5),
        dict(a=4, b=5, f=10),
    ]
)
y = np.array([0, 1], dtype=np.int64)
pipe.fit(data, y)

try:
    to_onnx(pipe, data[:1], options={"zipmap": False})
except Exception as e:
    print("It does not work:", e)

##################################
# Use of custom transformer
# +++++++++++++++++++++++++
#
# It is easier to write a custom converter if the FunctionTransformer
# is implemented as a custom transformer.


class OverpriceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def calculate_overprice(self, x, y):
        return 100 * (x - y) / y

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.apply(lambda x: self.calculate_overprice(x.a, x.b), axis=1)
        return x.values.reshape((-1, 1))


mapper = ColumnTransformer(
    transformers=[
        ("ab", FunctionTransformer(), ["a", "b"]),  # We keep the first column.
        ("c", OverpriceCalculator(), ["a", "b"]),  # We add a new one.
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

pipe_tr = Pipeline([("mapper", mapper), ("classifier", DecisionTreeClassifier())])
pipe_tr.fit(data, y)

#############################
# Both pipelines return the same output.
assert_allclose(pipe.predict_proba(data), pipe_tr.predict_proba(data))

#############################
# Let's check it produces the same number of features.
assert_allclose(pipe.steps[0][-1].transform(data), pipe_tr.steps[0][-1].transform(data))

#############################
# But the conversion still fails with a different error message.

try:
    to_onnx(pipe_tr, data[:1], options={"zipmap": False})
except Exception as e:
    print("It does not work:", e)


#################################
# Custom converter
# ++++++++++++++++
#
# We need to implement the method `calculate_overprice` in ONNX.
# The first function returns the expected type and shape.


def overprice_shape_calculator(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Gets the input type, the transformer works on any numerical type.
    input_type = operator.inputs[0].type.__class__
    # The first dimension is usually dynamic (batch dimension).
    input_dim = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = input_type([input_dim, 1])


def overprice_converter(scope, operator, container):
    # No need to retrieve the fitted estimator, it is not trained.
    # op = operator.raw_operator
    opv = container.target_opset
    X = operator.inputs[0]

    # 100 * (x-y)/y  --> 100 * (X[0] - X[1]) / X[1]

    zero = np.array([0], dtype=np.int64)
    one = np.array([1], dtype=np.int64)
    two = np.array([2], dtype=np.int64)
    hundred = np.array([100], dtype=np.float32)

    # Slice(data, starts, ends, axes)
    x0 = OnnxSlice(X, zero, one, one, op_version=opv)
    x1 = OnnxSlice(X, one, two, one, op_version=opv)
    z = OnnxMul(
        OnnxCastLike(hundred, X, op_version=opv),
        OnnxDiv(OnnxSub(x0, x1, op_version=opv), x1, op_version=opv),
        op_version=opv,
        output_names=operator.outputs[0],
    )
    z.add_to(scope, container)


update_registered_converter(
    OverpriceCalculator,
    "AliasOverpriceCalculator",
    overprice_shape_calculator,
    overprice_converter,
)


onx = to_onnx(pipe_tr, data[:1], target_opset=18, options={"zipmap": False})

############################
# Let's check there is no discrepancies
# +++++++++++++++++++++++++++++++++++++
#
# First the expected values

expected = (pipe_tr.predict(data), pipe_tr.predict_proba(data))
print(expected)

##############################
# Then let's check with :class:`onnx.reference.ReferenceEvaluator`.

feeds = {
    "a": data["a"].values.reshape((-1, 1)),
    "b": data["b"].values.reshape((-1, 1)),
    "f": data["f"].values.reshape((-1, 1)),
}

# verbose=10 to show intermediate results
ref = ReferenceEvaluator(onx, verbose=0)
got = ref.run(None, feeds)

assert_allclose(expected[0], got[0])
assert_allclose(expected[1], got[1])

#######################################
# Then with the runtime used to deploy, onnxruntime for example.

ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
got = ref.run(None, feeds)

assert_allclose(expected[0], got[0])
assert_allclose(expected[1], got[1])

#######################################
# Finally.
print("done")
