"""
Tests on functions in common.
"""

import unittest
from logging import getLogger
from distutils.version import StrictVersion
from pandas import DataFrame
from skl2onnx.validate import (
    enumerate_validated_operator_opsets, sklearn_operators,
    summary_report
)
import onnx


class TestValidate(unittest.TestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        assert len(res) > 0
        assert len(res[0]) == 3

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="OnnxOperator not working")
    def test_validate_sklearn_operators_one(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 0
        rows = list(enumerate_validated_operator_opsets(verbose, debug=False,
                    models={'NMF'}))
        assert len(rows) > 0
        df = DataFrame(rows)
        if verbose > 0:
            print("output results in Excel file")
        df.to_excel("sklearn_opsets_report_NMF.xlsx", index=False)
        assert df.shape[1] > 0
        sdf = summary_report(df)
        sdf.to_excel("sklearn_opsets_summary_NMF.xlsx", index=False)
        assert sdf.loc[0, 'Issue'] == "NO CONVERTER"

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="OnnxOperator not working")
    def test_validate_sklearn_operators_two(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 0
        rows = list(enumerate_validated_operator_opsets(verbose, debug=False,
                    models={'LinearRegression'}))
        assert len(rows) > 0
        assert rows[0]['max_abs_diff_batch'] <= 1e-5
        df = DataFrame(rows)
        if verbose > 0:
            print("output results in Excel file")
        df.to_excel("sklearn_opsets_report_LR.xlsx", index=False)
        assert df.shape[1] > 0
        sdf = summary_report(df)
        sdf.to_excel("sklearn_opsets_summary_LR.xlsx", index=False)
        assert sdf.loc[0, 'Opset'] == "1+"

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="OnnxOperator not working")
    def test_validate_sklearn_operators_all(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        verbose = 1 if __name__ == "__main__" else 0
        rows = list(enumerate_validated_operator_opsets(verbose, debug=False))
        assert len(rows) > 0
        df = DataFrame(rows)
        if verbose > 0:
            print("output results in Excel file")
        df.to_excel("sklearn_opsets_report.xlsx", index=False)
        assert df.shape[1] > 0
        sdf = summary_report(df)
        sdf.to_excel("sklearn_opsets_summary.xlsx", index=False)


if __name__ == "__main__":
    unittest.main()
