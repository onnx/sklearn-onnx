..  SPDX-License-Identifier: Apache-2.0


sklearn-onnx: Convert your scikit-learn model into ONNX
=======================================================

.. list-table:
    :header-rows: 1
    :widths: 5 5
    * - Linux
      - Windows
    * - .. image:: https://github.com/onnx/sklearn-onnx/actions/workflows/linux-ci.yml/badge.svg
            :target: https://github.com/onnx/sklearn-onnx/actions/workflows/linux-ci.yml
      - .. image:: https://github.com/onnx/sklearn-onnx/actions/workflows/windows-macos-ci.yml/badge.svg
            :target: https://github.com/onnx/sklearn-onnx/actions/workflows/windows-macos-ci.yml


*sklearn-onnx* enables you to convert models from
`scikit-learn <https://scikit-learn.org/stable/>`_
toolkits into `ONNX <https://onnx.ai>`_.

.. toctree::
    :maxdepth: 1

    introduction
    index_tutorial
    api_summary
    auto_examples/index
    pipeline
    parameterized
    supported

**Issues, questions**

You should look for `existing issues <https://github.com/onnx/sklearn-onnx/issues?utf8=%E2%9C%93&q=is%3Aissue>`_
or submit a new one. Sources are available on
`onnx/sklearn-onnx <https://github.com/onnx/sklearn-onnx>`_.

**ONNX version**

.. index:: target_opset, opset version

The converter can convert a model for a specific version of ONNX.
Every ONNX release is labelled with an opset number
returned by function `onnx_opset_version
<https://github.com/onnx/onnx/blob/main/onnx/defs/__init__.py#L22>`_.
This function returns the default value for parameter
target opset (parameter *target_opset*) if it is not specified
when converting the model. Every operator is versioned.
The library chooses the most recent version below or equal
to the targetted opset number for every operator.
The ONNX model has one opset number for every operator domain,
this value is the maximum opset number among all
onnx nodes.

.. runpython::
    :showcode:
    
    from skl2onnx import __max_supported_opset__, __version__
    print("documentation for version:", __version__)
    print("Last supported opset:", __max_supported_opset__)

**Backend**

*sklearn-onnx* converts models in ONNX format which
can be then used to compute predictions with the
backend of your choice. However, there exists a way
to automatically check every converter with
`onnxruntime <https://pypi.org/project/onnxruntime/>`_,
`onnxruntime-gpu <https://pypi.org/project/onnxruntime-gpu>`_.
Every converter is tested with this backend.

**Getting started**

::

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    X, y = iris.data, iris.target
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format.
    from skl2onnx import to_onnx

    onx = to_onnx(clr, X[:1])
    with open("rf_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # Compute the prediction with onnxruntime.
    import onnxruntime as rt

    sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]

**Related converters**

*sklearn-onnx* only converts models from *scikit-learn*.
`onnxmltools <https://github.com/onnx/onnxmltools>`_
can be used to convert models for *libsvm*, *lightgbm*, *xgboost*.
Other converters can be found on `github/onnx <https://github.com/onnx/>`_,
`torch.onnx <https://pytorch.org/docs/stable/onnx.html>`_,
`ONNX-MXNet API <https://mxnet.incubator.apache.org/api/python/contrib/onnx.html>`_,
`Microsoft.ML.Onnx <https://www.nuget.org/packages/Microsoft.ML.Onnx/>`_...

**Change Logs**

See `CHANGELOGS.md <https://github.com/onnx/sklearn-onnx/blob/main/CHANGELOGS.md>`_.

**Credits**

The package was started by the following engineers and data scientists at
Microsoft starting from winter 2017: Zeeshan Ahmed, Wei-Sheng Chin, Aidan Crook,
Xavier Dupré, Costin Eseanu, Tom Finley, Lixin Gong, Scott Inglis,
Pei Jiang, Ivan Matantsev, Prabhat Roy, M. Zeeshan Siddiqui,
Shouheng Yi, Shauheen Zahirazami, Yiwen Zhu, Du Li, Xuan Li, Wenbing Li.

**License**

It is licensed with `Apache License v2.0 <../LICENSE>`_.

**Older versions**

* `1.18.0 <versions/v1.18.0/>`_
* `1.17.0 <versions/v1.17.0/>`_
* `1.16.0 <versions/v1.16.0/>`_
