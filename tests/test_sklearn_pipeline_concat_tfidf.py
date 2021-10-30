# SPDX-License-Identifier: Apache-2.0

import unittest
import random
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import Fail
import pandas
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # not available in 0.19
    ColumnTransformer = None
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from skl2onnx import to_onnx
from test_utils import TARGET_OPSET


class TestSklearnPipelineConcatTfIdf(unittest.TestCase):

    words = ['ability', 'able', 'about', 'above', 'abroad',
             'absence', 'absolute', 'absolutely', 'absorb',
             'academic', 'accept', 'access', 'accident', 'accompany',
             'accomplish', 'according', 'account', 'accurate', 'achieve',
             'achievement', 'acid', 'acknowledge', 'acquire', 'across',
             'act', 'action', 'active', 'activity', 'actor', 'actress',
             'actual', 'actually', 'ad', 'adapt', 'add', 'addition',
             'additional', 'address', 'adequate', 'adjust',
             'adjustment', 'administration', 'administrator', 'admire',
             'admission', 'admit', 'adolescent', 'adopt', 'adult',
             'advance', 'advanced', 'advantage', 'adventure',
             'advice', 'advise', 'adviser', 'advocate', 'affair',
             'afford', 'afraid', 'after', 'afternoon', 'again', 'against',
             'age', 'agency', 'agenda', 'agent', 'aggressive', 'ago',
             'agree', 'agreement', 'agricultural', 'ah', 'ahead', 'aid',
             'aide', 'aim', 'air', 'aircraft', 'airline', 'airport',
             'alive', 'all', 'alliance', 'allow', 'ally', 'almost',
             'along', 'already', 'also', 'alter', 'alternative',
             'always', 'AM', 'amazing', 'among', 'amount', 'analysis',
             'analyze', 'ancient', 'and', 'anger', 'angle', 'angry',
             'anniversary', 'announce', 'annual', 'another', 'answer',
             'anticipate', 'anxiety', 'any', 'anybody', 'anymore',
             'anything', 'anyway', 'anywhere', 'apart', 'apartment',
             'apparently', 'appeal', 'appear', 'appearance', 'apple',
             'application', 'apply', 'appoint', 'appointment',
             'approach', 'appropriate', 'approval', 'approve',
             'architect', 'area', 'argue', 'argument', 'arise', 'arm',
             'around', 'arrange', 'arrangement', 'arrest',
             'arrival', 'arrive', 'art', 'article', 'artist', 'artistic',
             'as', 'aside', 'ask', 'asleep', 'aspect', 'assert',
             'assess', 'assessment', 'asset', 'assign', 'assignment',
             'assist', 'assistance', 'assistant', 'associate',
             'association', 'assume', 'assumption', 'assure', 'at',
             'athlete', 'athletic', 'atmosphere', 'attach', 'attack',
             'attempt', 'attend', 'attention', 'attitude', 'attract',
             'attractive', 'attribute', 'audience', 'author', 'auto',
             'available', 'average', 'avoid', 'award', 'aware',
             'away', 'baby', 'back', 'background', 'bag', 'bake',
             'balance', 'ball', 'band', 'bank', 'bar', 'barrel',
             'barrier', 'base', 'baseball', 'basic', 'basically',
             'a', 'to', 'the', 'an', 'than', 'of', 'off', 'us',
             'who', 'which', 'what', 'why', 'whom', 'at', 'from',
             'for', 'to', 'towards']

    @staticmethod
    def random_cats(n=10000, start=1000, end=9000):
        cats = numpy.random.randint(start, end, n)
        return numpy.array(["cat%d" % i for i in cats])

    @staticmethod
    def random_sentance(n=10000, length=7):
        words = TestSklearnPipelineConcatTfIdf.words
        ls = numpy.random.randint(1, length, n)
        text = []
        for size in ls:
            sentance = [words[random.randint(0, len(words)-1)]
                        for i in range(size)]
            text.append(" ".join(sentance))
        return numpy.array(text)

    @staticmethod
    def get_pipeline(N=10000):
        dfx = pandas.DataFrame(
            {'CAT1': TestSklearnPipelineConcatTfIdf.random_cats(N, 10, 20),
             'CAT2': TestSklearnPipelineConcatTfIdf.random_cats(N, 30, 40),
             'TEXT': TestSklearnPipelineConcatTfIdf.random_sentance(N)})
        dfy = numpy.random.randint(0, 2, N)

        dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(dfx, dfy)

        cat_features = ['CAT1', 'CAT2']
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore', sparse=True)
        textual_feature = 'TEXT'
        count_vect_transformer = Pipeline(steps=[
            ('count_vect', CountVectorizer(
                max_df=0.8, min_df=0.02, max_features=1000))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_transform', categorical_transformer,
                 cat_features),
                ('count_vector', count_vect_transformer,
                 textual_feature)])
        pipe = Pipeline(steps=[('preprocessor', preprocessor)])
        pipe.fit(dfx_train, dfy_train)
        dfx_test = dfx_test.reset_index(drop=True).copy()
        dfx_test.loc[0, 'TEXT'] = 'about'
        dfx_test.loc[1, 'TEXT'] = 'the'
        return pipe, dfx_test

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
    def test_issue_712_svc_binary(self):

        pipe, dfx_test = TestSklearnPipelineConcatTfIdf.get_pipeline()
        expected = pipe.transform(dfx_test)

        inputs = {'CAT1': dfx_test['CAT1'].values.reshape((-1, 1)),
                  'CAT2': dfx_test['CAT2'].values.reshape((-1, 1)),
                  'TEXT': dfx_test['TEXT'].values.reshape((-1, 1))}
        onx = to_onnx(pipe, dfx_test, target_opset=TARGET_OPSET)
        sess = InferenceSession(onx.SerializeToString())

        expected_dense = expected.todense()
        for i in range(dfx_test.shape[0]):
            row_inputs = {k: v[i: i+1] for k, v in inputs.items()}
            got = sess.run(None, row_inputs)
            assert_almost_equal(expected_dense[i], got[0])

        with self.assertRaises(Fail):
            # StringNormlizer removes empty strings after normalizer.
            # This case happens when a string contains only stopwords.
            # Then rows are missing and the output of the StringNormalizer
            # and the OneHotEncoder output cannot be merged anymore with
            # an error message like the following:
            #   onnxruntime.capi.onnxruntime_pybind11_state.Fail:
            #   [ONNXRuntimeError] : 1 : FAIL : Non-zero status code
            #   returned while running Concat node. Name:'Concat1'
            #   Status Message: concat.cc:159 onnxruntime::ConcatBase::
            #   PrepareForCompute Non concat axis dimensions must match:
            #   Axis 0 has mismatched dimensions of 2106 and 2500.
            got = sess.run(None, inputs)
        # assert_almost_equal(expected.todense(), got[0])

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available")
    @ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
    def test_issue_712_svc_binary_empty(self):

        pipe, dfx_test = TestSklearnPipelineConcatTfIdf.get_pipeline()
        expected = pipe.transform(dfx_test)

        inputs = {'CAT1': dfx_test['CAT1'].values.reshape((-1, 1)),
                  'CAT2': dfx_test['CAT2'].values.reshape((-1, 1)),
                  'TEXT': dfx_test['TEXT'].values.reshape((-1, 1))}
        onx = to_onnx(pipe, dfx_test, target_opset=TARGET_OPSET,
                      options={CountVectorizer: {'keep_empty_string': True}})
        with open("debug.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        sess = InferenceSession(onx.SerializeToString())

        expected_dense = expected.todense()
        for i in range(dfx_test.shape[0]):
            row_inputs = {k: v[i: i+1] for k, v in inputs.items()}
            got = sess.run(None, row_inputs)
            assert_almost_equal(expected_dense[i], got[0])

        got = sess.run(None, inputs)
        assert_almost_equal(expected.todense(), got[0])


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('skl2onnx')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    unittest.main()
