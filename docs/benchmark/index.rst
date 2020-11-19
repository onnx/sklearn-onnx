Benchmarks
==========

This page tracks graphs produced for publication.

2020/11
+++++++

*Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz, 2112 MHz, 4 cores, 8 logical processors*

::

    numpy==1.19.4
    scikit-learn==0.24.dev0
    onnx==1.8.1072
    onnxruntime==1.5.994
    skl2onnx==1.7.1092

.. image:: linear_model.png

.. image:: rf_model.png

.. image:: svm_model.png

To replicate:

::

    cd benchmark
    python -u bench_plot_onnxruntime_random_forest_reg.py
    python -u bench_plot_onnxruntime_svm_reg.py
    python -u bench_plot_onnxruntime_logreg.py
    python -u bench_plot_onnxruntime_linreg.py
    cd results
    python -u post_graph.py