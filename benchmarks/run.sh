python -u bench_plot_onnxruntime_linreg.py
python -u bench_plot_onnxruntime_logreg.py
python -u bench_plot_onnxruntime_svm_reg.py
python -u bench_plot_onnxruntime_random_forest_reg.py
cd results
python -u post_graph.py
