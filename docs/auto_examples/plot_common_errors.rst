.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_common_errors.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_common_errors.py:


.. _l-example-simple-usage:

Common errors with onnxruntime
==============================

This example looks into several common situations
in which *onnxruntime* does not return the model 
prediction but raises an exception instead.
It starts by loading a model
(see :ref:`l-rf-iris-example`).
which produces a logistic regression
trained on *Iris* datasets. The model takes
a vector of dimension 2 and returns a class among three.



.. code-block:: python

    import onnxruntime as rt
    import numpy
    from onnxruntime.datasets import get_example

    example2 = get_example("logreg_iris.onnx")
    sess = rt.InferenceSession(example2)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name







The first example fails due to *bad types*.
*onnxruntime* only expects single floats (4 bytes)
and cannot handle any other kind of floats.



.. code-block:: python


    try:
        x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float64)
        sess.run([output_name], {input_name: x})
    except Exception as e:
        print("Unexpected type")
        print("{0}: {1}".format(type(e), e))
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Unexpected type
    <class 'RuntimeError'>: Method run failed due to: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (class onnxruntime::NonOnnxType<double>) , expected: (class onnxruntime::NonOnnxType<float>)


The model fails to return an output if the name
is misspelled.



.. code-block:: python


    try:
        x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
        sess.run(["misspelled"], {input_name: x})
    except Exception as e:
        print("Misspelled output name")
        print("{0}: {1}".format(type(e), e))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Misspelled output name
    <class 'RuntimeError'>: Method run failed due to: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid Output Names: misspelled Valid output names are: label probabilities


The output name is optional, it can be replaced by *None*
and *onnxruntime* will then return all the outputs.



.. code-block:: python


    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
    res = sess.run(None, {input_name: x})
    print("All outputs")
    print(res)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    All outputs
    [array([0, 0, 0], dtype=int64), [{0: 0.950599730014801, 1: 0.027834169566631317, 2: 0.02156602405011654}, {0: 0.9974970817565918, 1: 5.6299926654901356e-05, 2: 0.0024466661270707846}, {0: 0.9997311234474182, 1: 1.1918064757310276e-07, 2: 0.00026869276189245284}]]


The same goes if the input name is misspelled.



.. code-block:: python


    try:
        x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
        sess.run([output_name], {"misspelled": x})
    except Exception as e:
        print("Misspelled input name")
        print("{0}: {1}".format(type(e), e))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Misspelled input name
    <class 'RuntimeError'>: Method run failed due to: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Missing required inputs: float_input


*onnxruntime* does not necessarily fail if the input
dimension is a multiple of the expected input dimension.



.. code-block:: python


    for x in [
            numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32),
            numpy.array([[1.0, 2.0, 3.0, 4.0]], dtype=numpy.float32),
            numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32),
            numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32),
            numpy.array([[1.0, 2.0, 3.0]], dtype=numpy.float32),
            ]:
        r = sess.run([output_name], {input_name: x})
        print("Shape={0} and predicted labels={1}".format(x.shape, r))

    for x in [
            numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32),
            numpy.array([[1.0, 2.0, 3.0, 4.0]], dtype=numpy.float32),
            numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32),
            numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32),
            numpy.array([[1.0, 2.0, 3.0]], dtype=numpy.float32),
            ]:
        r = sess.run(None, {input_name: x})
        print("Shape={0} and predicted probabilities={1}".format(x.shape, r[1]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Shape=(4,) and predicted labels=[array([2], dtype=int64)]
    Shape=(1, 4) and predicted labels=[array([2], dtype=int64)]
    Shape=(2, 2) and predicted labels=[array([0, 0], dtype=int64)]
    Shape=(3,) and predicted labels=[array([0], dtype=int64)]
    Shape=(1, 3) and predicted labels=[array([0], dtype=int64)]
    Shape=(4,) and predicted probabilities=[{0: 0.0009370420593768358, 1: 0.001740509644150734, 2: 0.9973224401473999}]
    Shape=(1, 4) and predicted probabilities=[{0: 0.0009370420593768358, 1: 0.001740509644150734, 2: 0.9973224401473999}]
    Shape=(2, 2) and predicted probabilities=[{0: 0.950599730014801, 1: 0.027834169566631317, 2: 0.02156602405011654}, {0: 0.9974970817565918, 1: 5.6299926654901356e-05, 2: 0.0024466661270707846}]
    Shape=(3,) and predicted probabilities=[{0: 0.7892322540283203, 1: 0.20707039535045624, 2: 0.0036973499227315187}]
    Shape=(1, 3) and predicted probabilities=[{0: 0.7892322540283203, 1: 0.20707039535045624, 2: 0.0036973499227315187}]


It does not fail either if the number of dimension
is higher than expects but produces a warning.



.. code-block:: python


    for x in [
            numpy.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=numpy.float32),
            numpy.array([[[1.0, 2.0, 3.0]]], dtype=numpy.float32),
            numpy.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=numpy.float32),
            ]:
        r = sess.run([output_name], {input_name: x})
        print("Shape={0} and predicted labels={1}".format(x.shape, r))




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Shape=(1, 2, 2) and predicted labels=[array([0], dtype=int64)]
    Shape=(1, 1, 3) and predicted labels=[array([1], dtype=int64)]
    Shape=(2, 1, 2) and predicted labels=[array([1, 1], dtype=int64)]


**Total running time of the script:** ( 0 minutes  0.030 seconds)


.. _sphx_glr_download_auto_examples_plot_common_errors.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_common_errors.py <plot_common_errors.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_common_errors.ipynb <plot_common_errors.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
