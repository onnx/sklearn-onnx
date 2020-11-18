@echo off

rem @echo START > bench_plot_onnxruntime_linreg.py.txt
rem python -u bench_plot_onnxruntime_linreg.py
rem @echo END > bench_plot_onnxruntime_linreg.py.txt

@echo START > bench_plot_onnxruntime_logreg.py.txt
python -u bench_plot_onnxruntime_logreg.py
@echo END > bench_plot_onnxruntime_logreg.py.txt

@echo START > bench_plot_onnxruntime_svm_reg.py.txt
python -u bench_plot_onnxruntime_svm_reg.py
@echo END > bench_plot_onnxruntime_svm_reg.py.txt

@echo START > bench_plot_onnxruntime_random_forest_reg.py.txt
python -u bench_plot_onnxruntime_random_forest_reg.py
@echo END > bench_plot_onnxruntime_random_forest_reg.py.txt
