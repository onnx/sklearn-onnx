#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#-------------------------------------------------------------------------
"""
You can run this file with to get a report on every tested model conversion.

::

    python -u tests/benchmark.py <folder>

Folder contains the model to compare implemented as unit tests.
"""

import os
import sys
import unittest
import warnings
import platform
import cpuinfo



def run_all_tests(folder=None):
    """
    Runs all unit tests or unit tests specific to one library.
    The tests produce a series of files dumped into ``folder``
    which can be later used to tests a backend (or a runtime).
    
    :param folder: where to put the dumped files
    """
    if folder is None:
        folder = 'TESTDUMP'
    os.environ["ONNXTESTDUMP"] = folder
    os.environ["ONNXTESTDUMPERROR"] = "1"
    os.environ["ONNXTESTBENCHMARK"] = "1"

    try:
        import onnxmltools
    except ImportError:
        raise ImportError("Cannot import onnxmltools. It must be installed first.")
    
    this = os.path.abspath(os.path.dirname(__file__))
    subs = [this]
    loader = unittest.TestLoader()
    suites = []
    
    for sub in subs:
        fold = os.path.join(this, sub)
        if not os.path.exists(fold):
            raise FileNotFoundError("Unable to find '{0}'".format(fold))
        
        # ts = loader.discover(fold)
        sys.path.append(fold)
        names = [_ for _ in os.listdir(fold) if _.startswith("test")]
        for name in names:
            name = os.path.splitext(name)[0]
            ts = loader.loadTestsFromName(name)
            suites.append(ts)
        index = sys.path.index(fold)
        del sys.path[index]
    
    with warnings.catch_warnings():
        warnings.filterwarnings(category=DeprecationWarning, action="ignore")
        warnings.filterwarnings(category=FutureWarning, action="ignore")
        runner = unittest.TextTestRunner()
        for tsi, ts in enumerate(suites):
            for k in ts:
                for t in k:
                    print(t.__class__.__name__)
                    break            
            runner.run(ts)
    
    from onnxmltools.utils.tests_helper import make_report_backend
    report = make_report_backend(folder)
    
    from pandas import DataFrame, set_option
    set_option("display.max_columns", None)
    set_option("display.max_rows", None)
    
    df = DataFrame(report).sort_values(["_model"])
    
    import onnx
    import onnxruntime
    print(df)
    df["onnx-version"] = onnx.__version__
    df["onnxruntime-version"] = onnxruntime.__version__
    cols = list(df.columns)
    if 'stderr' in cols:
        ind = cols.index('stderr')
        del cols[ind]
        cols += ['stderr']
        df = df[cols]
    df["ratio"] = df["onnxrt_time"] / df["original_time"]
    df["CPU"] = platform.processor()
    df["CPUI"] = cpuinfo.get_cpu_info()['brand']
    df.to_excel(os.path.join(folder, "report_backend.xlsx"))
    return df
                    
    
if __name__ == "__main__":
    folder = None if len(sys.argv) < 2 else sys.argv[1]
    run_all_tests(folder=folder)
