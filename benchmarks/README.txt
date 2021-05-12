To run the benchmark:

All benchmarks produces csv files written in subfolder *results*.
Benchmark can be run the following way:

::

    python bench_plot_onnxruntime_linreg.py
    python bench_plot_onnxruntime_logreg.py
    python bench_plot_onnxruntime_random_forest_reg.py
    python bench_plot_onnxruntime_svm_reg.py

In subfolder *results*, script post_graph produces
graph for each of them.

::

    python results/post_graph.py

