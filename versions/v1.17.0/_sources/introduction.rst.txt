..  SPDX-License-Identifier: Apache-2.0


============
Introduction
============

Quick start
===========

*ONNX Runtime* provides an easy way to run
machine learned models with high performance on CPU or GPU
without dependencies on the training framework.
Machine learning frameworks are usually optimized for
batch training rather than for prediction, which is a
more common scenario in applications, sites, and services.
At a high level, you can:

1. Train a model using your favorite framework.
2. Convert or export the model into ONNX format.
   See `ONNX Tutorials <https://github.com/onnx/tutorials>`_
   for more details.
3. Load and run the model using *ONNX Runtime*.

In this tutorial, we will briefly create a
pipeline with *scikit-learn*, convert it into
ONNX format and run the first predictions.

Step 1: Train a model using your favorite framework
+++++++++++++++++++++++++++++++++++++++++++++++++++

We'll use the famous Iris datasets.

::

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.linear_model import LogisticRegression
    clr = LogisticRegression()
    clr.fit(X_train, y_train)

Step 2: Convert or export the model into ONNX format
++++++++++++++++++++++++++++++++++++++++++++++++++++

`ONNX <https://github.com/onnx/onnx>`_ is a format to describe
the machine learned model.
It defines a set of commonly used operators to compose models.
There are `tools <https://github.com/onnx/tutorials>`_
to convert other model formats into ONNX. Here we will use
`ONNXMLTools <https://github.com/onnx/onnxmltools>`_.

::

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    with open("logreg_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())

Step 3: Load and run the model using ONNX Runtime
+++++++++++++++++++++++++++++++++++++++++++++++++

We will use *ONNX Runtime* to compute the predictions
for this machine learning model.

::

    import onnxruntime as rt
    sess = rt.InferenceSession("logreg_iris.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

.. index:: initial types

convert_sklearn, to_onnx, initial_types
=======================================

The module implements two functions:
:func:`convert_sklearn <skl2onnx.convert_sklearn>` and
:func:`to_onnx <skl2onnx.to_onnx>`. The first one
was used in the previous examples, it requires two
mandatory arguments:

* a *scikit-learn* model or a pipeline
* initial types

*scikit-learn* does not store information about
the training dataset. It is not always possible to retrieve
the number of features or their types. That's why the
function needs another argument called *initial_types*.
In many cases, the training datasets is a numerical matrix
*X_train*. Then it becomes
``initial_type=[('X', FloatTensorType([None, X_train.shape[1]]))]``.
*X* is the name of this unique input, the second term indicates the
type and shape. The shape is ``[None, X_train.shape[1]]``,
the first dimension is the number of rows followed by the
number of features. The number of rows is undefined as the
the number of requested predictions is unknown at the time
the model is converted. The number of features is usually known.
Let's assume now the input is a string column followed by
a matrix, then initial types would be:

::

    initial_type=[
        ('S', StringTensorType([None, 1])),
        ('X', FloatTensorType([None, X_train.shape[1]])),
    ]

Function :func:`to_onnx <skl2onnx.to_onnx>` was implemented
after discussions with the core developers of *scikit-learn*.
It also contains a mechanism to infer the proper type based on
one row of the training datasets. Then, the following code
``convert_sklearn(clr, initial_types=[('X', FloatTensorType([None, 4]))])``
is usually rewritten into ``to_onnx(clr, X_train[:1])`` where
*X_train* is the training dataset, it can be a matrix or a
dataframe. The input name is ``'X'`` by default unless *X_train*
is a dataframe. In that case, the column names are used
as input names.
