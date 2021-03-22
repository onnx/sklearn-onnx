# SPDX-License-Identifier: Apache-2.0

"""
@brief      test log(time=3s)
"""

import unittest
from skl2onnx.tutorial.imagenet_classes import class_names


class TestUtilsClasses(unittest.TestCase):

    def test_classes(self):
        cl = class_names
        self.assertIsInstance(cl, dict)
        self.assertEqual(len(cl), 1000)


if __name__ == "__main__":
    unittest.main()
