import unittest
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from skl2onnx import convert_sklearn
from test_utils import dump_multiple_classification, dump_binary_classification


class TestVotingClassifierConverter(unittest.TestCase):

    def test_voting_hard_binary(self):
        model = VotingClassifier(voting='hard', estimators=[
                    ('lr', LogisticRegression()),
                    ('lr2', LogisticRegression(fit_intercept=False))])
        # predict_proba is not defined when voting is hard.
        dump_binary_classification(model, suffix='Hard-OneOffArray',
                                   comparable_outputs=[0,])

    def test_voting_soft_binary(self):
        model = VotingClassifier(voting='soft', estimators=[
                    ('lr', LogisticRegression()),
                    ('lr2', LogisticRegression(fit_intercept=False))])
        dump_binary_classification(model, suffix='Soft-OneOffArray',
                                   comparable_outputs=[0, 1])

    def test_voting_hard_multi(self):
        # predict_proba is not defined when voting is hard.
        model = VotingClassifier(voting='soft', estimators=[
                    ('lr', LogisticRegression()),
                    ('lr2', DecisionTreeClassifier())])
        dump_multiple_classification(model, suffix='Hard-OneOffArray')

    def test_voting_soft_multi(self):
        model = VotingClassifier(voting='soft', estimators=[
                    ('lr', LogisticRegression()),
                    ('lr2', LogisticRegression())])
        dump_multiple_classification(model, suffix='Soft-OneOffArray')


if __name__ == "__main__":
    unittest.main()
