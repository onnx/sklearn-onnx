"""
.. _l-plot-optim-tree-ensemble:

TreeEnsemble optimization
=========================

::

    export PYTHONPATH=~/github/onnxruntime/build/linux_cpu/Release/:$PYTHONPATH

"""

import logging
import os
import timeit
from typing import Tuple
import numpy
import onnx
from onnx.helper import make_graph, make_model
from onnx.reference import ReferenceEvaluator
from pandas import DataFrame
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx
from onnxruntime import InferenceSession, SessionOptions
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from onnx_extended.reference import CReferenceEvaluator
from onnx_extended.ortops.optim.cpu import get_ort_ext_libs
from onnx_extended.ortops.optim.optimize import optimize_model
from onnx_extended.tools.onnx_nodes import multiply_tree
from onnx_extended.args import get_parsed_args
from onnx_extended.ext_test_case import unit_test_going
from onnx_extended.plotting.benchmark import hhistograms

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

script_args = get_parsed_args(
    "plot_benchmark_tree",
    description=__doc__,
    scenarios={
        "SHORT": "short optimization (default)",
        "LONG": "test more options",
        "CUSTOM": "use values specified by the command line",
    },
    n_features=(2 if unit_test_going() else 5, "number of features to generate"),
    n_trees=(3 if unit_test_going() else 10, "number of trees to train"),
    max_depth=(2 if unit_test_going() else 5, "max_depth"),
    batch_size=(1000 if unit_test_going() else 10000, "batch size"),
    parallel_tree=("80,160,40", "values to try for parallel_tree"),
    parallel_tree_N=("256,128,64", "values to try for parallel_tree_N"),
    parallel_N=("100,50,25", "values to try for parallel_N"),
    expose="",
    n_jobs=("-1", "number of jobs to train the RandomForestRegressor"),
)


################################
# Training a model
# ++++++++++++++++


def train_model(
    batch_size: int, n_features: int, n_trees: int, max_depth: int
) -> Tuple[str, numpy.ndarray, numpy.ndarray]:
    filename = f"plot_op_tree_ensemble_optim-f{n_features}-{n_trees}-d{max_depth}.onnx"
    if not os.path.exists(filename):
        X, y = make_regression(
            batch_size + max(batch_size, 2 ** (max_depth + 1)),
            n_features=n_features,
            n_targets=1,
        )
        print(f"Training to get {filename!r} with X.shape={X.shape}")
        X, y = X.astype(numpy.float32), y.astype(numpy.float32)
        # To be faster, we train only 1 tree.
        model = RandomForestRegressor(
            1, max_depth=max_depth, verbose=2, n_jobs=int(script_args.n_jobs)
        )
        model.fit(X[:-batch_size], y[:-batch_size])
        onx = to_onnx(model, X[:1], target_opset={"": 18, "ai.onnx.ml": 3})

        # And wd multiply the trees.
        node = multiply_tree(onx.graph.node[0], n_trees)
        onx = make_model(
            make_graph([node], onx.graph.name, onx.graph.input, onx.graph.output),
            domain=onx.domain,
            opset_imports=onx.opset_import,
            ir_version=onx.ir_version,
        )

        with open(filename, "wb") as f:
            f.write(onx.SerializeToString())
    else:
        X, y = make_regression(batch_size, n_features=n_features, n_targets=1)
        X, y = X.astype(numpy.float32), y.astype(numpy.float32)
    Xb, yb = X[-batch_size:].copy(), y[-batch_size:].copy()
    return filename, Xb, yb


batch_size = script_args.batch_size
n_features = script_args.n_features
n_trees = script_args.n_trees
max_depth = script_args.max_depth

print(f"batch_size={batch_size}")
print(f"n_features={n_features}")
print(f"n_trees={n_trees}")
print(f"max_depth={max_depth}")

##############################
# training

filename, Xb, yb = train_model(batch_size, n_features, n_trees, max_depth)

print(f"Xb.shape={Xb.shape}")
print(f"yb.shape={yb.shape}")

#######################################
# Rewrite the onnx file to use a different kernel
# +++++++++++++++++++++++++++++++++++++++++++++++
#
# The custom kernel is mapped to a custom operator with the same name
# the attributes and domain = `"onnx_extended.ortops.optim.cpu"`.
# We call a function to do that replacement.
# First the current model.

with open(filename, "rb") as f:
    onx = onnx.load(f)
print(onnx_simple_text_plot(onx))

############################
# And then the modified model.


def transform_model(
    model: onnx.ModelProto,
    parallel_tree: int,
    parallel_tree_N: int,
    parallel_N: int,
    first=-556,
):
    """
    Modifies the graph.
    Attributes is unused ``nodes_hitrates_as_tensor`` by the runtime so we use
    that field to specify parallelization settings.

    :param model: model proto serialized, the function makes a copy
    :param parallel_tree: see https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/ml/tree_ensemble_common.h#L38
    :param parallel_tree_N: see https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/ml/tree_ensemble_common.h#L39
    :param parallel_N: see https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/ml/tree_ensemble_common.h#L40
    :param first: -556 or -555, -556 makes onnxruntime prints out the parallelization
        settings to make sure these are the expected values
    :return: ModelProto
    """
    onx = onnx.ModelProto()
    onx.ParseFromString(model.SerializeToString())
    new_nodes = []
    for node in onx.graph.node:
        if node.op_type.startswith("TreeEnsemble"):
            new_atts = []
            for att in node.attribute:
                if att.name.startswith("nodes_hitrates"):
                    continue
                new_atts.append(att)
            new_atts.append(
                onnx.helper.make_attribute(
                    "nodes_hitrates_as_tensor",
                    onnx.numpy_helper.from_array(
                        numpy.array(
                            [first, parallel_tree, parallel_tree_N, parallel_N],
                            dtype=numpy.float32,
                        ),
                        name="nodes_hitrates_as_tensor",
                    ),
                )
            )
            del node.attribute[:]
            node.attribute.extend(new_atts)
            new_nodes.append(node)
            continue
        new_nodes.append(node)
    del onx.graph.node[:]
    onx.graph.node.extend(new_nodes)
    del onx.opset_import[:]
    onx.opset_import.extend(
        [onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("ai.onnx.ml", 3)]
    )
    return onx


print("Tranform model to add a custom node.")
onx_modified = transform_model(onx, 777, 778, 779)
print(f"Save into {filename + 'modified.onnx'!r}.")
with open(filename + "modified.onnx", "wb") as f:
    f.write(onx_modified.SerializeToString())
print("done.")
print(onnx_simple_text_plot(onx_modified))

#######################################
# Comparing onnxruntime and the custom kernel
# +++++++++++++++++++++++++++++++++++++++++++

print(f"Loading {filename!r}")
sess_ort = InferenceSession(filename, providers=["CPUExecutionProvider"])

r = get_ort_ext_libs()
print(f"Creating SessionOptions with {r!r}")
opts = SessionOptions()
if r is not None:
    opts.register_custom_ops_library(r[0])

print(f"Loading modified {filename!r}")
sess_cus = InferenceSession(
    onx_modified.SerializeToString(), opts, providers=["CPUExecutionProvider"]
)

print(f"Running once with shape {Xb.shape}.")
base = sess_ort.run(None, {"X": Xb})[0]
print(f"Running modified with shape {Xb.shape}.")
got = sess_cus.run(None, {"X": Xb})[0]
print("done.")

#######################################
# Discrepancies?

d = numpy.abs(base - got)
ya = numpy.abs(base).mean()
print(f"Discrepancies: max={d.max() / ya}, mean={d.mean() / ya} (A={ya})")

########################################
# Simple verification
# +++++++++++++++++++
#
# Baseline with onnxruntime.
t1 = timeit.timeit(lambda: sess_ort.run(None, {"X": Xb}), number=50)
print(f"baseline: {t1}")

#################################
# The custom implementation.
t2 = timeit.timeit(lambda: sess_cus.run(None, {"X": Xb}), number=50)
print(f"new time: {t2}")

#################################
# The same implementation but ran from the onnx python backend.
ref = CReferenceEvaluator(filename)
ref.run(None, {"X": Xb})
t3 = timeit.timeit(lambda: ref.run(None, {"X": Xb}), number=50)
print(f"CReferenceEvaluator: {t3}")

#################################
# The python implementation but from the onnx python backend.
if n_trees < 50:
    # It is usully slow.
    ref = ReferenceEvaluator(filename)
    ref.run(None, {"X": Xb})
    t4 = timeit.timeit(lambda: ref.run(None, {"X": Xb}), number=5)
    print(f"ReferenceEvaluator: {t4} (only 5 times instead of 50)")


#############################################
# Time for comparison
# +++++++++++++++++++
#
# The custom kernel supports the same attributes as *TreeEnsembleRegressor*
# plus new ones to tune the parallelization. They can be seen in
# `tree_ensemble.cc <https://github.com/sdpython/onnx-extended/
# blob/main/onnx_extended/ortops/optim/cpu/tree_ensemble.cc#L102>`_.
# Let's try out many possibilities.
# The default values are the first ones.

if unit_test_going():
    optim_params = dict(
        parallel_tree=[40],  # default is 80
        parallel_tree_N=[128],  # default is 128
        parallel_N=[50, 25],  # default is 50
    )
elif script_args.scenario in (None, "SHORT"):
    optim_params = dict(
        parallel_tree=[80, 40],  # default is 80
        parallel_tree_N=[128, 64],  # default is 128
        parallel_N=[50, 25],  # default is 50
    )
elif script_args.scenario == "LONG":
    optim_params = dict(
        parallel_tree=[80, 160, 40],
        parallel_tree_N=[256, 128, 64],
        parallel_N=[100, 50, 25],
    )
elif script_args.scenario == "CUSTOM":
    optim_params = dict(
        parallel_tree=[int(i) for i in script_args.parallel_tree.split(",")],
        parallel_tree_N=[int(i) for i in script_args.parallel_tree_N.split(",")],
        parallel_N=[int(i) for i in script_args.parallel_N.split(",")],
    )
else:
    raise ValueError(
        f"Unknown scenario {script_args.scenario!r}, use --help to get them."
    )

cmds = []
for att, value in optim_params.items():
    cmds.append(f"--{att}={','.join(map(str, value))}")
print("Full list of optimization parameters:")
print(" ".join(cmds))

##################################
# Then the optimization.


def create_session(onx):
    return InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])


res = optimize_model(
    onx,
    feeds={"X": Xb},
    transform=transform_model,
    session=create_session,
    baseline=lambda onx: InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    ),
    params=optim_params,
    verbose=True,
    number=script_args.number,
    repeat=script_args.repeat,
    warmup=script_args.warmup,
    sleep=script_args.sleep,
    n_tries=script_args.tries,
)

###############################
# And the results.

df = DataFrame(res)
df.to_csv("plot_op_tree_ensemble_optim.csv", index=False)
df.to_excel("plot_op_tree_ensemble_optim.xlsx", index=False)
print(df.columns)
print(df.head(5))

################################
# Sorting
# +++++++

small_df = df.drop(
    [
        "min_exec",
        "max_exec",
        "repeat",
        "number",
        "context_size",
        "n_exp_name",
    ],
    axis=1,
).sort_values("average")
print(small_df.head(n=10))


################################
# Worst
# +++++

print(small_df.tail(n=10))


#################################
# Plot
# ++++

skeys = ",".join(optim_params.keys())
title = f"TreeEnsemble tuning, n_tries={script_args.tries}\n{skeys}\nlower is better"
ax = hhistograms(df, title=title, keys=("name",))
fig = ax.get_figure()
fig.savefig("plot_op_tree_ensemble_optim.png")
