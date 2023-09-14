..  SPDX-License-Identifier: Apache-2.0


Tutorial
========

.. index:: tutorial

The tutorial goes from a simple example which
converts a pipeline to a more complex example
involving operator not actually implemented in
:epkg:`ONNX operators` or :epkg:`ONNX ML operators`.

.. toctree::
    :maxdepth: 2

    tutorial_1_simple
    tutorial_1-5_external
    tutorial_2_new_converter
    tutorial_4_advanced
    tutorial_2-5_extlib

The tutorial was tested with following version:

.. runpython::
    :showcode:

    import catboost
    import numpy
    import scipy
    import sklearn
    import lightgbm
    import onnx
    import onnxmltools
    import onnxruntime
    import xgboost
    import skl2onnx

    mods = [numpy, scipy, sklearn, lightgbm, xgboost, catboost,
            onnx, onnxmltools, onnxruntime,
            skl2onnx]
    mods = [(m.__name__, m.__version__) for m in mods]
    mx = max(len(_[0]) for _ in mods) + 1
    for name, vers in sorted(mods):
        print("%s%s%s" % (name, " " * (mx - len(name)), vers))
