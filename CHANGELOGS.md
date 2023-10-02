# Change Logs

## 1.16.0

* fix OneHotEncoder when categories indices to drop are not None
  [#1028](https://github.com/onnx/sklearn-onnx/pull/1028)
* fix converter for AdaBoost estimators in scikit-learn==1.3.1
  [#1027](https://github.com/onnx/sklearn-onnx/pull/1027)
* add function 'add_onnx_graph' to insert onnx graph coming from other converting
  libraries within the converter mapped to a custom estimator
  [#1023](https://github.com/onnx/sklearn-onnx/pull/1023)
* add option 'language' to converters of CountVectorizer, TfIdfVectorizer
  [#1020](https://github.com/onnx/sklearn-onnx/pull/1020)
