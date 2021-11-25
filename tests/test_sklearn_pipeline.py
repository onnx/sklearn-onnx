# SPDX-License-Identifier: Apache-2.0

import unittest
import urllib.error as url_error
from distutils.version import StrictVersion
from io import StringIO
import warnings
import numpy
from numpy.testing import assert_almost_equal
import pandas
from sklearn import __version__ as skl_version
from sklearn import datasets
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import (
        make_column_transformer, make_column_selector)
except ImportError:
    # not available in 0.19
    ColumnTransformer = None
    make_column_selector = None
    make_column_transformer = None
from sklearn.decomposition import PCA, TruncatedSVD

try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler,
    MaxAbsScaler)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from sklearn.multioutput import MultiOutputClassifier
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model, fit_classification_model, TARGET_OPSET)
from onnxruntime import __version__ as ort_version, InferenceSession


# StrictVersion does not work with development versions
ort_version = ".".join(ort_version.split('.')[:2])
skl_version = ".".join(skl_version.split('.')[:2])


def check_scikit_version():
    return StrictVersion(skl_version) >= StrictVersion("0.22")


class PipeConcatenateInput:
    def __init__(self, pipe):
        self.pipe = pipe

    def transform(self, inp):
        if isinstance(inp, (numpy.ndarray, pandas.DataFrame)):
            return self.pipe.transform(inp)
        elif isinstance(inp, dict):
            keys = list(sorted(inp.keys()))
            dim = inp[keys[0]].shape[0], len(keys)
            x2 = numpy.zeros(dim)
            for i in range(x2.shape[1]):
                x2[:, i] = inp[keys[i]].ravel()
            res = self.pipe.transform(x2)
            return res
        else:
            raise TypeError("Unable to predict with type {0}".format(
                type(inp)))


class TestSklearnPipeline(unittest.TestCase):

    @ignore_warnings(category=FutureWarning)
    def test_pipeline(self):
        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]],
                           dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        model_onnx = convert_sklearn(model, "pipeline",
                                     [("input", FloatTensorType([None, 2]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx,
                            basename="SklearnPipelineScaler")

    @ignore_warnings(category=FutureWarning)
    def test_combine_inputs(self):
        data = numpy.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        model_onnx = convert_sklearn(
            model,
            "pipeline",
            [
                ("input1", FloatTensorType([None, 1])),
                ("input2", FloatTensorType([None, 1])),
            ],
            target_opset=TARGET_OPSET)
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = {
            "input1": data[:, 0].reshape((-1, 1)),
            "input2": data[:, 1].reshape((-1, 1)),
        }
        dump_data_and_model(
            data, PipeConcatenateInput(model),
            model_onnx, basename="SklearnPipelineScaler11")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=FutureWarning)
    def test_combine_inputs_union_in_pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = numpy.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=numpy.float32,
        )
        model = Pipeline([
            ("scaler1", StandardScaler()),
            (
                "union",
                FeatureUnion([
                    ("scaler2", StandardScaler()),
                    ("scaler3", MinMaxScaler()),
                ]),
            ),
        ])
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "pipeline",
            [
                ("input1", FloatTensorType([None, 1])),
                ("input2", FloatTensorType([None, 1])),
            ],
            target_opset=TARGET_OPSET)
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = {
            "input1": data[:, 0].reshape((-1, 1)),
            "input2": data[:, 1].reshape((-1, 1)),
        }
        dump_data_and_model(
            data, PipeConcatenateInput(model),
            model_onnx, basename="SklearnPipelineScaler11Union")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=FutureWarning)
    def test_combine_inputs_floats_ints(self):
        data = [[0, 0.0], [0, 0.0], [1, 1.0], [1, 1.0]]
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        model_onnx = convert_sklearn(
            model,
            "pipeline",
            [
                ("input1", Int64TensorType([None, 1])),
                ("input2", FloatTensorType([None, 1])),
            ],
            target_opset=TARGET_OPSET)
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = numpy.array(data)
        data = {
            "input1": data[:, 0].reshape((-1, 1)).astype(numpy.int64),
            "input2": data[:, 1].reshape((-1, 1)).astype(numpy.float32),
        }
        dump_data_and_model(
            data, PipeConcatenateInput(model),
            model_onnx, basename="SklearnPipelineScalerMixed")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion("0.4.0"),
                     reason="issues with shapes")
    @ignore_warnings(category=(RuntimeWarning, FutureWarning))
    def test_pipeline_column_transformer(self):

        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: "cat1"
                                              if x > 0.5 else "cat2")
        X_train["vcat2"] = X_train["vB"].apply(lambda x: "cat3"
                                               if x > 0.5 else "cat4")
        y_train = y % 2
        numeric_features = [0, 1, 2]  # ["vA", "vB", "vC"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01,
            class_weight=dict(zip([False, True], [0.2, 0.8])),
            n_jobs=1, max_iter=10, solver="lbfgs", tol=1e-3)

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            (
                "onehot",
                OneHotEncoder(sparse=True, handle_unknown="ignore"),
            ),
            (
                "tsvd",
                TruncatedSVD(n_components=1, algorithm="arpack", tol=1e-4),
            ),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])

        model = Pipeline(steps=[("precprocessor",
                                 preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)
        initial_type = [
            ("numfeat", FloatTensorType([None, 3])),
            ("strfeat", StringTensorType([None, 2])),
        ]

        X_train = X_train[:11]
        model_onnx = convert_sklearn(model, initial_types=initial_type,
                                     target_opset=TARGET_OPSET)

        dump_data_and_model(
            X_train, model, model_onnx,
            basename="SklearnPipelineColumnTransformerPipeliner")

        if __name__ == "__main__":
            from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

            pydot_graph = GetPydotGraph(
                model_onnx.graph,
                name=model_onnx.graph.name,
                rankdir="TP",
                node_producer=GetOpNodeProducer("docstring"))
            pydot_graph.write_dot("graph.dot")

            import os

            os.system("dot -O -G=300 -Tpng graph.dot")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.20 causes some mismatches")
    @ignore_warnings(category=FutureWarning)
    def test_pipeline_column_transformer_titanic(self):

        # fit
        try:
            titanic_url = (
                "https://raw.githubusercontent.com/amueller/"
                "scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv")
            data = pandas.read_csv(titanic_url)
        except url_error.URLError:
            # Do not fail the test if the data cannot be fetched.
            warnings.warn("Unable to fetch titanic data.")
            return
        X = data.drop("survived", axis=1)
        y = data["survived"]

        # SimpleImputer on string is not available for string
        # in ONNX-ML specifications.
        # So we do it beforehand.
        for cat in ["embarked", "sex", "pclass"]:
            X[cat].fillna("missing", inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        numeric_features = ["age", "fare"]
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_features = ["embarked", "sex", "pclass"]
        categorical_transformer = Pipeline(steps=[
            # --- SimpleImputer on string is not available
            # for string in ONNX-ML specifications.
            # ('imputer',
            #  SimpleImputer(strategy='constant', fill_value='missing')),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])

        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            # ("classifier", LogisticRegression(solver="lbfgs")),
        ])

        # inputs

        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == 'int64':
                    t = Int64TensorType([None, 1])
                elif v == "float64":
                    t = FloatTensorType([None, 1])
                else:
                    t = StringTensorType([None, 1])
                inputs.append((k, t))
            return inputs

        to_drop = {
            "parch",
            "sibsp",
            "cabin",
            "ticket",
            "name",
            "body",
            "home.dest",
            "boat",
        }

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train['pclass'] = X_train['pclass'].astype(numpy.int64)
        X_test['pclass'] = X_test['pclass'].astype(numpy.int64)
        X_train = X_train.drop(to_drop, axis=1)
        X_test = X_test.drop(to_drop, axis=1)

        # Step 1: without classifier
        clf.fit(X_train, y_train)
        initial_inputs = convert_dataframe_schema(X_train, to_drop)
        model_onnx = convert_sklearn(clf, "pipeline_titanic", initial_inputs,
                                     target_opset=TARGET_OPSET)

        data = X_test
        pred = clf.transform(data)
        data_types = {
            'pclass': numpy.int64,
            'age': numpy.float32,
            'sex': numpy.str_,
            'fare': numpy.float32,
            'embarked': numpy.str_,
        }
        inputs = {k: data[k].values.astype(data_types[k]).reshape(-1, 1)
                  for k in data.columns}
        sess = InferenceSession(model_onnx.SerializeToString())
        run = sess.run(None, inputs)
        got = run[-1]
        assert_almost_equal(pred, got, decimal=5)

        # Step 2: with classifier
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="lbfgs")),
        ]).fit(X_train, y_train)
        pred = clf.predict_proba(data)
        model_onnx = convert_sklearn(clf, "pipeline_titanic", initial_inputs,
                                     target_opset=TARGET_OPSET,
                                     options={id(clf): {'zipmap': False}})
        sess = InferenceSession(model_onnx.SerializeToString())
        run = sess.run(None, inputs)
        got = run[-1]
        assert_almost_equal(pred, got, decimal=5)

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_weights(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(10, 100))],
                transformer_weights={'pca': 2, 'svd': 3}), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer weights",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerWeights-Dec4")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_drop(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(80, 100))],
                remainder='drop'), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer drop",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerDrop")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_passthrough(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(80, 100))],
                transformer_weights={'pca': 2, 'svd': 3},
                remainder='passthrough'), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer passthrough",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerPassthrough")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_passthrough_no_weights(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(70, 80))],
                remainder='passthrough'), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer passthrough",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerPassthroughNoWeights")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @ignore_warnings(category=FutureWarning)
    def test_pipeline_dataframe(self):
        text = """
                fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,color
                7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
                7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,red
                7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,red
                11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,red
                """.replace("                ", "")
        X_train = pandas.read_csv(StringIO(text))
        for c in X_train.columns:
            if c != 'color':
                X_train[c] = X_train[c].astype(numpy.float32)
        numeric_features = [c for c in X_train if c != 'color']

        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("color", Pipeline([
                    ('one', OneHotEncoder()),
                    ('select', ColumnTransformer(
                        [('sel1', 'passthrough', [0])]))
                ]), ['color']),
                ("others", "passthrough", numeric_features)
            ])),
        ])

        init_types = [
            ('fixed_acidity', FloatTensorType(shape=[None, 1])),
            ('volatile_acidity', FloatTensorType(shape=[None, 1])),
            ('citric_acid', FloatTensorType(shape=[None, 1])),
            ('residual_sugar', FloatTensorType(shape=[None, 1])),
            ('chlorides', FloatTensorType(shape=[None, 1])),
            ('free_sulfur_dioxide', FloatTensorType(shape=[None, 1])),
            ('total_sulfur_dioxide', FloatTensorType(shape=[None, 1])),
            ('density', FloatTensorType(shape=[None, 1])),
            ('pH', FloatTensorType(shape=[None, 1])),
            ('sulphates', FloatTensorType(shape=[None, 1])),
            ('alcohol', FloatTensorType(shape=[None, 1])),
            ('quality', FloatTensorType(shape=[None, 1])),
            ('color', StringTensorType(shape=[None, 1]))
        ]

        pipe.fit(X_train)
        model_onnx = convert_sklearn(
            pipe, initial_types=init_types, target_opset=TARGET_OPSET)
        oinf = InferenceSession(model_onnx.SerializeToString())

        pred = pipe.transform(X_train)
        inputs = {c: X_train[c].values for c in X_train.columns}
        inputs = {c: v.reshape((v.shape[0], 1)) for c, v in inputs.items()}
        onxp = oinf.run(None, inputs)
        got = onxp[0]
        assert_almost_equal(pred, got)

    @ignore_warnings(category=(FutureWarning, UserWarning))
    def test_pipeline_tfidf_svc(self):
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf_svc', SVC(probability=True, kernel='linear'))])
        data = numpy.array(["first sentance", "second sentence",
                            "many sentances", "dummy sentance",
                            "no sentance at all"])
        y = numpy.array([0, 0, 1, 0, 1])
        pipe.fit(data, y)
        expected_label = pipe.predict(data)
        expected_proba = pipe.predict_proba(data)
        df = pandas.DataFrame(data)
        df.columns = ['text']

        # first conversion if shape=[None, 1]
        model_onnx = convert_sklearn(
            pipe, initial_types=[('text', StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
            options={id(pipe): {'zipmap': False}})
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'text': data.reshape((-1, 1))})
        assert_almost_equal(expected_proba, got[1])
        assert_almost_equal(expected_label, got[0])
        # sess.run(None, {'text': df}) --> failures
        # sess.run(None, {'text': df["text"]}) --> failures

        # second conversion with shape=[None]
        model_onnx = convert_sklearn(
            pipe, initial_types=[('text', StringTensorType([None]))],
            target_opset=TARGET_OPSET,
            options={id(pipe): {'zipmap': False}})
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'text': data})
        assert_almost_equal(expected_proba, got[1])
        assert_almost_equal(expected_label, got[0])
        # sess.run(None, {'text': df})  failure
        # sess.run(None, {'text': df["text"]})  failure
        sess.run(None, {'text': df["text"].values})  # success

    @ignore_warnings(category=(FutureWarning, UserWarning))
    def test_pipeline_voting_tfidf_svc(self):
        pipe1 = Pipeline([
            ('tfidf1', TfidfVectorizer()),
            ('svc', SVC(probability=True, kernel='linear'))])
        pipe2 = Pipeline([
            ('tfidf2', TfidfVectorizer(norm='l2', use_idf=False)),
            ('sgd', SGDClassifier(alpha=0.0001, penalty='l2',
                                  loss='modified_huber'))])
        pipe3 = Pipeline([
            ('tfidf3', TfidfVectorizer()),
            ('mnb', MultinomialNB())])
        voting = VotingClassifier(
            [('p1', pipe1), ('p2', pipe2), ('p3', pipe3)],
            voting='soft', flatten_transform=False)
        data = numpy.array(["first sentance", "second sentence",
                            "many sentances", "dummy sentance",
                            "no sentance at all"])
        y = numpy.array([0, 0, 1, 0, 1])
        voting.fit(data, y)
        expected_label = voting.predict(data)
        expected_proba = voting.predict_proba(data)
        df = pandas.DataFrame(data)
        df.columns = ['text']

        model_onnx = convert_sklearn(
            voting, initial_types=[('text', StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
            options={id(voting): {'zipmap': False}})
        # with open("debug.onnx", "wb") as f:
        #     f.write(model_onnx.SerializeToString())
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'text': data.reshape((-1, 1))})
        assert_almost_equal(expected_proba, got[1], decimal=5)
        assert_almost_equal(expected_label, got[0])

    @ignore_warnings(category=(FutureWarning, UserWarning))
    def test_pipeline_pipeline_voting_tfidf_svc(self):
        pipe1 = Pipeline([
            ('ntfidf1', Pipeline([
                ('tfidf1', TfidfVectorizer()),
                ('scaler', FeatureUnion([
                    ('scaler2', StandardScaler(with_mean=False)),
                    ('mm', MaxAbsScaler())]))])),
            ('svc', SVC(probability=True, kernel='linear'))])
        pipe2 = Pipeline([
            ('tfidf2', TfidfVectorizer(norm='l2', use_idf=False)),
            ('sgd', SGDClassifier(alpha=0.0001, penalty='l2',
                                  loss='modified_huber'))])
        pipe3 = Pipeline([
            ('tfidf3', TfidfVectorizer()),
            ('mnb', MultinomialNB())])
        voting = VotingClassifier(
            [('p1', pipe1), ('p2', pipe2), ('p3', pipe3)],
            voting='soft', flatten_transform=False)
        data = numpy.array(["first sentance", "second sentence",
                            "many sentances", "dummy sentance",
                            "no sentance at all"])
        y = numpy.array([0, 0, 1, 0, 1])
        voting.fit(data, y)
        expected_label = voting.predict(data)
        expected_proba = voting.predict_proba(data)
        df = pandas.DataFrame(data)
        df.columns = ['text']

        model_onnx = convert_sklearn(
            voting, initial_types=[('text', StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
            options={id(voting): {'zipmap': False}})
        # with open("debug.onnx", "wb") as f:
        #     f.write(model_onnx.SerializeToString())
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'text': data.reshape((-1, 1))})
        assert_almost_equal(expected_proba, got[1])
        assert_almost_equal(expected_label, got[0])

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.21 too old")
    @ignore_warnings(category=(FutureWarning, UserWarning))
    def test_pipeline_pipeline_rf(self):
        cat_feat = ['A', 'B']
        text_feat = 'TEXT'

        pipe = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('cat_tr', OneHotEncoder(handle_unknown='ignore'),
                     cat_feat),
                    ('count_vect', Pipeline(steps=[
                        ('count_vect', CountVectorizer(
                            max_df=0.8, min_df=0.05, max_features=1000))]),
                     text_feat)])),
            ('classifier', MultiOutputClassifier(
                estimator=RandomForestClassifier(
                    n_estimators=5, max_depth=5)))])

        data = numpy.array([
            ["cat1", "cat2", "cat3", "cat1", "cat2"],
            ["C1", "C2", "C3", "C3", "C4"],
            ["first sentance", "second sentence",
             "many sentances", "dummy sentance",
             "no sentance at all"]]).T
        y = numpy.array([[0, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
        df = pandas.DataFrame(data, columns=['A', 'B', 'TEXT'])
        pipe.fit(df, y)
        expected_label = pipe.predict(df)
        expected_proba = pipe.predict_proba(df)

        model_onnx = convert_sklearn(
            pipe, initial_types=[
                ('A', StringTensorType([None, 1])),
                ('B', StringTensorType([None, 1])),
                ('TEXT', StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
            options={MultiOutputClassifier: {'zipmap': False}})
        # with open("debug.onnx", "wb") as f:
        #     f.write(model_onnx.SerializeToString())
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'A': data[:, :1], 'B': data[:, 1:2],
                              'TEXT': data[:, 2:]})
        self.assertEqual(len(expected_proba), len(got[1]))
        for e, g in zip(expected_proba, got[1]):
            assert_almost_equal(e, g, decimal=5)
        assert_almost_equal(expected_label, got[0])

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.21 too old")
    @ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
    def test_issue_712_multio(self):
        dfx = pandas.DataFrame(
            {'CAT1': ['985332', '985333', '985334', '985335', '985336'],
             'CAT2': ['1985332', '1985333', '1985334', '1985335', '1985336'],
             'TEXT': ["abc abc", "abc def", "def ghj", "abcdef", "abc ii"]})
        dfy = pandas.DataFrame(
            {'REAL': [5, 6, 7, 6, 5],
             'CATY': [0, 1, 0, 1, 0]})

        cat_features = ['CAT1', 'CAT2']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        textual_feature = 'TEXT'
        count_vect_transformer = Pipeline(steps=[
            ('count_vect', CountVectorizer(
                max_df=0.8, min_df=0.05, max_features=1000))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_transform', categorical_transformer, cat_features),
                ('count_vector', count_vect_transformer, textual_feature)])
        model_RF = RandomForestClassifier(random_state=42, max_depth=50)
        rf_clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MultiOutputClassifier(estimator=model_RF))])
        rf_clf.fit(dfx, dfy)
        expected_label = rf_clf.predict(dfx)
        expected_proba = rf_clf.predict_proba(dfx)

        inputs = {'CAT1': dfx['CAT1'].values.reshape((-1, 1)),
                  'CAT2': dfx['CAT2'].values.reshape((-1, 1)),
                  'TEXT': dfx['TEXT'].values.reshape((-1, 1))}
        onx = to_onnx(rf_clf, dfx, target_opset=TARGET_OPSET,
                      options={MultiOutputClassifier: {'zipmap': False}})
        sess = InferenceSession(onx.SerializeToString())

        got = sess.run(None, inputs)
        assert_almost_equal(expected_label, got[0])
        self.assertEqual(len(expected_proba), len(got[1]))
        for e, g in zip(expected_proba, got[1]):
            assert_almost_equal(e, g, decimal=5)

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.21 too old")
    @ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
    def test_issue_712_svc_multio(self):
        for sub_model in [LinearSVC(), SVC()]:
            for method in ["sigmoid", "isotonic"]:
                with self.subTest(sub_model=sub_model, method=method):
                    dfx = pandas.DataFrame(
                        {'CAT1': ['985332', '985333', '985334', '985335',
                                  '985336', '985332', '985333', '985334',
                                  '985335', '985336', '985336'],
                         'CAT2': ['1985332', '1985333', '1985334', '1985335',
                                  '1985336', '1985332', '1985333', '1985334',
                                  '1985335', '1985336', '1985336'],
                         'TEXT': ["abc abc", "abc def", "def ghj", "abcdef",
                                  "abc ii", "abc abc", "abc def", "def ghj",
                                  "abcdef", "abc ii", "abc abc"]})
                    dfy = pandas.DataFrame(
                        {'REAL': [5, 6, 7, 6, 5, 5, 6, 7, 5, 6, 7],
                         'CATY': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})

                    cat_features = ['CAT1', 'CAT2']
                    categorical_transformer = OneHotEncoder(
                        handle_unknown='ignore')
                    textual_feature = 'TEXT'
                    count_vect_transformer = Pipeline(steps=[
                        ('count_vect', CountVectorizer(
                            max_df=0.8, min_df=0.05, max_features=1000))])
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat_transform', categorical_transformer,
                             cat_features),
                            ('count_vector', count_vect_transformer,
                             textual_feature)])
                    model_SVC = CalibratedClassifierCV(
                        sub_model, cv=2, method=method)
                    rf_clf = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', MultiOutputClassifier(
                            estimator=model_SVC))])
                    rf_clf.fit(dfx, dfy)
                    expected_label = rf_clf.predict(dfx)
                    expected_proba = rf_clf.predict_proba(dfx)

                    inputs = {'CAT1': dfx['CAT1'].values.reshape((-1, 1)),
                              'CAT2': dfx['CAT2'].values.reshape((-1, 1)),
                              'TEXT': dfx['TEXT'].values.reshape((-1, 1))}
                    onx = to_onnx(
                        rf_clf, dfx, target_opset=TARGET_OPSET,
                        options={MultiOutputClassifier: {'zipmap': False}})
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, inputs)
                    assert_almost_equal(expected_label, got[0])
                    self.assertEqual(len(expected_proba), len(got[1]))
                    for e, g in zip(expected_proba, got[1]):
                        if method == "isotonic" and isinstance(sub_model, SVC):
                            # float/double issues
                            assert_almost_equal(e[2:4], g[2:4], decimal=3)
                        else:
                            assert_almost_equal(e, g, decimal=5)

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.21 too old")
    @ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
    def test_issue_712_svc_binary0(self):
        for sub_model in [LinearSVC(), SVC()]:
            for method in ["sigmoid", "isotonic"]:
                with self.subTest(sub_model=sub_model, method=method):
                    dfx = pandas.DataFrame(
                        {'CAT1': ['985332', '985333', '985334', '985335',
                                  '985336', '985332', '985333', '985334',
                                  '985335', '985336', '985336'],
                         'CAT2': ['1985332', '1985333', '1985334', '1985335',
                                  '1985336', '1985332', '1985333', '1985334',
                                  '1985335', '1985336', '1985336'],
                         'TEXT': ["abc abc", "abc def", "def ghj", "abcdef",
                                  "abc ii", "abc abc", "abc def", "def ghj",
                                  "abcdef", "abc ii", "abc abc"]})
                    dfy = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

                    cat_features = ['CAT1', 'CAT2']
                    categorical_transformer = OneHotEncoder(
                        handle_unknown='ignore')
                    textual_feature = 'TEXT'
                    count_vect_transformer = Pipeline(steps=[
                        ('count_vect', CountVectorizer(
                            max_df=0.8, min_df=0.05, max_features=1000))])
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat_transform', categorical_transformer,
                             cat_features),
                            ('count_vector', count_vect_transformer,
                             textual_feature)])
                    model_SVC = CalibratedClassifierCV(
                        sub_model, cv=2, method=method)
                    rf_clf = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', model_SVC)])
                    rf_clf.fit(dfx, dfy)
                    expected_label = rf_clf.predict(dfx)
                    expected_proba = rf_clf.predict_proba(dfx)

                    inputs = {'CAT1': dfx['CAT1'].values.reshape((-1, 1)),
                              'CAT2': dfx['CAT2'].values.reshape((-1, 1)),
                              'TEXT': dfx['TEXT'].values.reshape((-1, 1))}
                    onx = to_onnx(rf_clf, dfx, target_opset=TARGET_OPSET,
                                  options={'zipmap': False})
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, inputs)
                    assert_almost_equal(expected_label, got[0])
                    assert_almost_equal(expected_proba, got[1], decimal=5)

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.21 too old")
    @ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
    def test_issue_712_svc_multi(self):
        for sub_model in [SVC(), LinearSVC()]:
            for method in ["isotonic", "sigmoid"]:
                with self.subTest(sub_model=sub_model, method=method):
                    dfx = pandas.DataFrame(
                        {'CAT1': ['985332', '985333', '985334', '985335',
                                  '985336', '985332', '985333', '985334',
                                  '985335', '985336', '985336'],
                         'CAT2': ['1985332', '1985333', '1985334', '1985335',
                                  '1985336', '1985332', '1985333', '1985334',
                                  '1985335', '1985336', '1985336'],
                         'TEXT': ["abc abc", "abc def", "def ghj", "abcdef",
                                  "abc ii", "abc abc", "abc def", "def ghj",
                                  "abcdef", "abc ii", "abc abc"]})
                    dfy = numpy.array([5, 6, 7, 6, 5, 5, 8, 7, 5, 6, 8])

                    cat_features = ['CAT1', 'CAT2']
                    categorical_transformer = OneHotEncoder(
                        handle_unknown='ignore')
                    textual_feature = 'TEXT'
                    count_vect_transformer = Pipeline(steps=[
                        ('count_vect', CountVectorizer(
                            max_df=0.8, min_df=0.05, max_features=1000))])
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat_transform', categorical_transformer,
                             cat_features),
                            ('count_vector', count_vect_transformer,
                             textual_feature)])
                    model_SVC = CalibratedClassifierCV(
                        sub_model, cv=2, method=method)
                    rf_clf = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', model_SVC)])
                    rf_clf.fit(dfx, dfy)
                    expected_label = rf_clf.predict(dfx)
                    expected_proba = rf_clf.predict_proba(dfx)

                    inputs = {'CAT1': dfx['CAT1'].values.reshape((-1, 1)),
                              'CAT2': dfx['CAT2'].values.reshape((-1, 1)),
                              'TEXT': dfx['TEXT'].values.reshape((-1, 1))}
                    onx = to_onnx(rf_clf, dfx, target_opset=TARGET_OPSET,
                                  options={'zipmap': False})
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, inputs)
                    assert_almost_equal(expected_label, got[0])
                    if method == "isotonic":
                        # float/double issues
                        assert_almost_equal(
                            expected_proba[2:4], got[1][2:4], decimal=3)
                    else:
                        assert_almost_equal(expected_proba, got[1], decimal=5)

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.21 too old")
    @ignore_warnings(category=(FutureWarning, UserWarning))
    def test_pipeline_make_column_selector(self):
        X = pandas.DataFrame({
            'city': ['London', 'London', 'Paris', 'Sallisaw'],
            'rating': [5, 3, 4, 5]})
        X['rating'] = X['rating'].astype(numpy.float32)
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(
                dtype_include=numpy.number)),
            (OneHotEncoder(), make_column_selector(
                dtype_include=object)))
        expected = ct.fit_transform(X)
        onx = to_onnx(ct, X, target_opset=TARGET_OPSET)
        sess = InferenceSession(onx.SerializeToString())
        names = [i.name for i in sess.get_inputs()]
        got = sess.run(None, {names[0]: X[names[0]].values.reshape((-1, 1)),
                              names[1]: X[names[1]].values.reshape((-1, 1))})
        assert_almost_equal(expected, got[0])

    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.21 too old")
    def test_feature_selector_no_converter(self):

        class ColumnSelector(TransformerMixin, BaseEstimator):
            def __init__(self, cols):
                if not isinstance(cols, list):
                    self.cols = [cols]
                else:
                    self.cols = cols

            def fit(self, X, y):
                return self

            def transform(self, X):
                X = X.copy()
                return X[self.cols]

        # Inspired from
        # https://github.com/databricks/automl/blob/main/
        # runtime/tests/automl_runtime/sklearn/column_selector_test.py
        X_in = pandas.DataFrame(
            numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        dtype=numpy.float32),
            columns=["a", "b", "c"])
        y = pandas.DataFrame(numpy.array([[1], [0], [1]]),
                             columns=["label"])
        X_out_expected = numpy.array([1, 0, 1])

        standardizer = StandardScaler()
        selected_cols = ["a", "b"]
        col_selector = ColumnSelector(selected_cols)
        preprocessor = ColumnTransformer(
            [("standardizer", standardizer, selected_cols)], remainder="drop")

        model = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("decision_tree", DecisionTreeClassifier())
        ])
        model.fit(X=X_in, y=y)
        # Add one column so that the dataframe for prediction is
        # different with the data for training
        X_in["useless"] = 1
        X_out = model.predict(X_in)
        assert_almost_equal(X_out, X_out_expected)

        with self.assertRaises(RuntimeError) as e:
            to_onnx(model, X_in)
            self.assertIn('ColumnTransformer', str(e))


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('skl2onnx')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    unittest.main()
