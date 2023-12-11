# Change Logs

## 1.16.0

* Supports cosine distance (LocalOutlierFactor, ...)
  [#1050](https://github.com/onnx/sklearn-onnx/pull/1050),
* Add an example on how to handle FunctionTransformer
  [#1042](https://github.com/onnx/sklearn-onnx/pull/1042),
  Versions of `scikit-learn < 1.0` are not tested any more.
* FeatureHasher, raise an error when the delimiter length is > 1,
  [#1036](https://github.com/onnx/sklearn-onnx/pull/1036)
* skl2onnx works with onnx==1.15.0,
  [#1034](https://github.com/onnx/sklearn-onnx/pull/1034)
* fix OneHotEncoder when categories indices to drop are not None
  [#1028](https://github.com/onnx/sklearn-onnx/pull/1028)
* fix converter for AdaBoost estimators in scikit-learn==1.3.1
  [#1027](https://github.com/onnx/sklearn-onnx/pull/1027)
* add function 'add_onnx_graph' to insert onnx graph coming from other converting,  
  libraries within the converter mapped to a custom estimator
  [#1023](https://github.com/onnx/sklearn-onnx/pull/1023),
  [#1024](https://github.com/onnx/sklearn-onnx/pull/1024)
* add option 'language' to converters of CountVectorizer, TfIdfVectorizer
  [#1020](https://github.com/onnx/sklearn-onnx/pull/1020)
