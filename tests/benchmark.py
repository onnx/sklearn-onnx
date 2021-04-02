# SPDX-License-Identifier: Apache-2.0

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


def run_all_tests(folder=None, verbose=True):
    """
    Runs all unit tests or unit tests specific to one library.
    The tests produce a series of files dumped into ``folder``
    which can be later used to tests a backend (or a runtime).

    :param folder: where to put the dumped files
    :param verbose: verbose
    """
    if folder is None:
        folder = "TESTDUMP"
    os.environ["ONNXTESTDUMP"] = folder
    os.environ["ONNXTESTDUMPERROR"] = "1"
    os.environ["ONNXTESTBENCHMARK"] = "1"

    if verbose:
        print("[benchmark] look into '{0}'".format(folder))

    try:
        import onnxmltools  # noqa
    except ImportError:
        warnings.warn("Cannot import onnxmltools. Some tests won't work.")

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
                try:
                    for t in k:
                        print(t.__class__.__name__)
                        break
                except TypeError as e:
                    raise RuntimeError(
                        "Unable to run test '{}'.".format(ts)) from e
            runner.run(ts)

    from test_utils.tests_helper import make_report_backend

    df = make_report_backend(folder, as_df=True)

    from pandas import set_option

    set_option("display.max_columns", None)
    set_option("display.max_rows", None)
    exfile = os.path.join(folder, "report_backend.xlsx")
    df.to_excel(exfile)
    if verbose:
        print("[benchmark] wrote report in '{0}'".format(exfile))
    return df


if __name__ == "__main__":
    folder = None if len(sys.argv) < 2 else sys.argv[1]
    run_all_tests(folder=folder)
