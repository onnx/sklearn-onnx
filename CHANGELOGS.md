# Change Logs

## 1.20.0

* Fixes unknown_value=np.nan in OrdinalEncoder
  [#1198](https://github.com/onnx/sklearn-onnx/issues/1198)

## 1.19.1

* Fix QDA converter crashing on string labels and incorrect shape calculation
  [#1169](https://github.com/onnx/sklearn-onnx/issues/1169)
* Remove import of split_complex_to_pairs
  [#1182](https://github.com/onnx/sklearn-onnx/issues/1182)
* Removes dependency on onnxconverter-common
  [#1179](https://github.com/onnx/sklearn-onnx/issues/1179)
* Implements converter for QuantileTransformer
  [#1164](https://github.com/onnx/sklearn-onnx/issues/1164),
* Refactors KNNImputer converter with local function to match
  scikit-learn's implementation, the code was partially
  automatically generated from an equivalent implementation
  in pytorch and exported into ONNX
  [#1167](https://github.com/onnx/sklearn-onnx/issues/1167),
  [#1165](https://github.com/onnx/sklearn-onnx/issues/1165)
* Add support to sklearn TargetEncoder
  [#1137](https://github.com/onnx/sklearn-onnx/issues/1137)
* Fixes missing WhiteKernel with return_std=True #1163
  [#1163](https://github.com/onnx/sklearn-onnx/issues/1163)
* Fix empty column selector
  [#1159](https://github.com/onnx/sklearn-onnx/issues/1159)
* Fix conversion for XGBClassifier and XGBRegressor
  [#1157](https://github.com/onnx/sklearn-onnx/issues/1157)
* Test SelectKBest + StandardScaler pipeline
  [#1156](https://github.com/onnx/sklearn-onnx/issues/1156)
* Fix np.NAN into np.nan,
  [#1148](https://github.com/onnx/sklearn-onnx/issues/1148)

## 1.18.0

* Converter for OneHotEncoder does not add a concat operator if not needed,
  [#1110](https://github.com/onnx/sklearn-onnx/pull/1110)
* Function ``to_onnx`` now forces the main opset to be equal to the
  value speficied by the user (parameter ``target_opset``),
  [#1109](https://github.com/onnx/sklearn-onnx/pull/1109)
* Add converter for TunedThresholdClassifierCV,
  [#1107](https://github.com/onnx/sklearn-onnx/pull/1107)
* Update and Fix documentation
  [#1113](https://github.com/onnx/sklearn-onnx/pull/1113)
* Support fill_value for SimpleImputer with string data
  [#1123](https://github.com/onnx/sklearn-onnx/pull/1123)
* Remove unnecessary options for Regressor
  [#1124](https://github.com/onnx/sklearn-onnx/pull/1124)
* OrdinalEncoder handle encoded_missing_value and unknown_value
  [#1132](https://github.com/onnx/sklearn-onnx/pull/1132)
* Create output_onnx_single_probability.py
  [#1139](https://github.com/onnx/sklearn-onnx/pull/1139),
  [#1141](https://github.com/onnx/sklearn-onnx/pull/1141)

## 1.17.0 (development)

* Upgrade the maximum supported opset to 21,
  update requirements to scikit-learn>=1.1,
  older versions are not tested anymore,
  [#1098](https://github.com/onnx/sklearn-onnx/pull/1098)
* Support infrequent categories for OneHotEncoder
  [#1029](https://github.com/onnx/sklearn-onnx/pull/1029)
* Support kernel Matern in Gaussian Process
  [#978](https://github.com/onnx/sklearn-onnx/pull/978)
* Fix for multidimensional gaussian process
  [#1097](https://github.com/onnx/sklearn-onnx/pull/1097)
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
