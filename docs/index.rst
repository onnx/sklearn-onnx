
scikit-onnx: Convert your scikit-learn model into ONNX
======================================================

.. list-table:
    :header-rows: 1
    :widths: 5 5
    * - Linux
      - Windows
    * - .. image:: https://dev.azure.com/onnxmltools/sklearn-onnx/_apis/build/status/sklearn-onnx-linux-conda-ci?branchName=master
            :target: https://dev.azure.com/onnxmltools/sklearn-onnx/_build/latest?definitionId=5?branchName=master
      - .. image:: https://dev.azure.com/onnxmltools/sklearn-onnx/_apis/build/status/sklearn-onnx-win32-conda-ci?branchName=master
            :target: https://dev.azure.com/onnxmltools/sklearn-onnx/_build/latest?definitionId=5?branchName=master
	    
	    
*scikit-onnx* enables you to convert models from
`scikit-learn <https://scikit-learn.org/stable/>`_
toolkits into `ONNX <https://onnx.ai>`_. 

*scikit-onnx* converts models in ONNX format which
can be then used to compute predictions with the
backend of your choice. However, there exists a way
to automatically check every converter with
`onnxruntime <https://pypi.org/project/onnxruntime/>`_,
`onnxruntime-gpu <https://pypi.org/project/onnxruntime-gpu>`_.

::

    # Train a model.
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format with onnxmltools
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

    # Compute the prediction with ONNX Runtime
    import onnxruntime as rt
    import numpy
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

.. toctree::
    :maxdepth: 2
    
    tutorial
    api_summary
    auto_examples/index
    tests

If you want the converted model is compatible with certain ONNX version,
please specify the *target_opset* parameter on invoking convert function,
and the following Keras converter example code shows how it works.

The package was developed by the following engineers and data scientists at 
Microsoft starting from winter 2017: Zeeshan Ahmed, Wei-Sheng Chin, Aidan Crook, 
Xavier Dupré, Costin Eseanu, Tom Finley, Lixin Gong, Scott Inglis, 
Pei Jiang, Ivan Matantsev, Prabhat Roy, M. Zeeshan Siddiqui, 
Shouheng Yi, Shauheen Zahirazami, Yiwen Zhu, Du Li, Xuan Li, Wenbing Li.
It is licensed with `MIT License <../LICENSE>`_.


