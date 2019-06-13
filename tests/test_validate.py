"""
Tests on functions in common.
"""

import unittest
from logging import getLogger
from distutils.version import StrictVersion
from pandas import DataFrame
from skl2onnx.validate import sklearn_operators, validate_operator_opsets
import onnx


class TestValidate(unittest.TestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        assert len(res) > 0
        assert len(res[0]) == 4

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="OnnxOperator not working")
    def test_validate_sklearn_operators_all(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = validate_operator_opsets(verbose, debug=None)
        # debug={"DecisionTreeRegressor"})
        assert len(rows) > 0
        df = DataFrame(rows)
        if verbose > 0:
            print("output results in Excel file")
            df.to_excel("sklearn_opsets_report.xlsx", index=False)
            df.to_csv("sklearn_opsets_report.csv", index=False)
        assert df.shape[1] > 0


if __name__ == "__main__":
    unittest.main()
