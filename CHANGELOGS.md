# Change Logs

## 1.17.0 (development)

* Supports infrequent category with OneHotEncoder
  [#1029](https://github.com/onnx/sklearn-onnx/pull/1029)
* Minor fixes to support scikit-learn==1.5.0
  [#1095](https://github.com/onnx/sklearn-onnx/pull/1095)
* Fix the conversion of pipeline including pipelines,
  issue [#1069](https://github.com/onnx/sklearn-onnx/pull/1069),
  [#1072](https://github.com/onnx/sklearn-onnx/pull/1072)
* Fix unexpected type for intercept in PoissonRegressor and GammaRegressor
  [#1070](https://github.com/onnx/sklearn-onnx/pull/1070)
* Add support for scikit-learn 1.4.0,
  [#1058](https://github.com/onnx/sklearn-onnx/pull/1058),
  fixes issues [Many examples in the gallery are showing "broken"](https://github.com/onnx/sklearn-onnx/pull/1057),
  [TFIDF vectorizer target_opset issue](https://github.com/onnx/sklearn-onnx/pull/1055),
  [Tfidfvectorizer with sublinear_tf fails, despite opset version set to greater than 11](https://github.com/onnx/sklearn-onnx/pull/996).

## 1.16.0

* Supports cosine distance (LocalOutlierFactor, ...)
  [#1050](https://github.com/onnx/sklearn-onnx/pull/1050),
* Supports multiple columns for OrdinalEncoder
  [#1044](https://github.com/onnx/sklearn-onnx/pull/1044) (by @max-509)
* Add an example on how to handle FunctionTransformer
  [#1042](https://github.com/onnx/sklearn-onnx/pull/1042),
  Versions of `scikit-learn < 1.0` are not tested any more.
* Supports lists of strings as inputs for FeatureHasher
  [#1025](https://github.com/onnx/sklearn-onnx/pull/1036),
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
