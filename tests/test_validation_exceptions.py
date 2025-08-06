# SPDX-License-Identifier: Apache-2.0


import unittest
from skl2onnx.common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
    InvalidInputLengthException,
    InvalidOutputLengthException,
    InvalidInputTypeException,
    InvalidOutputTypeException,
)
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from skl2onnx.common._topology import Scope


class TestValidationExceptions(unittest.TestCase):
    def setUp(self):
        # Create a real scope
        self.scope = Scope("test_scope", target_opset=12)

        # Create some real variables
        self.float_var1 = self.scope.declare_local_variable(
            "float_input_1", FloatTensorType([None, 1])
        )
        self.float_var2 = self.scope.declare_local_variable(
            "float_input_2", FloatTensorType([None, 2])
        )
        self.int_var1 = self.scope.declare_local_variable(
            "int_input_1", Int64TensorType([None, 1])
        )
        self.string_var1 = self.scope.declare_local_variable(
            "string_output_1", StringTensorType([None, 1])
        )

        # Create a real operator, doesn't matter which one,
        # we'll use a simple linear model type
        self.operator = self.scope.declare_local_operator("LinearRegressor")

        # Manually set up inputs and outputs for testing
        self.operator.inputs.append(self.float_var1)
        self.operator.outputs.append(self.int_var1)

    def test_invalid_output_length_exception_too_few(self):
        # Operator has 1 output but we require at least 2
        with self.assertRaises(InvalidOutputLengthException) as cm:
            check_input_and_output_numbers(
                self.operator, input_count_range=1, output_count_range=[2, None]
            )

        exc = cm.exception
        self.assertEqual(exc.operator, self.operator)
        self.assertEqual(exc.min_output_count, 2)
        self.assertIsNone(exc.max_output_count)
        self.assertIn("at least 2 output(s)", str(exc))
        self.assertIn("LinearRegressor", str(exc))

    def test_invalid_output_length_exception_too_many(self):
        # Create operator with 3 outputs
        operator_with_many_outputs = self.scope.declare_local_operator("TestOp")
        operator_with_many_outputs.inputs.append(self.float_var1)
        operator_with_many_outputs.outputs.extend(
            [
                self.scope.declare_local_variable(
                    "float_input_N1", FloatTensorType([None, 2])
                ),
                self.scope.declare_local_variable(
                    "float_input_N2", FloatTensorType([None, 3])
                ),
                self.scope.declare_local_variable(
                    "float_input_N3", FloatTensorType([None, 3])
                ),
            ]
        )

        # Require at most 2 outputs
        with self.assertRaises(InvalidOutputLengthException) as cm:
            check_input_and_output_numbers(
                operator_with_many_outputs,
                input_count_range=1,
                output_count_range=[1, 2],
            )

        exc = cm.exception
        self.assertEqual(exc.operator, operator_with_many_outputs)
        self.assertIsNone(exc.min_output_count)
        self.assertEqual(exc.max_output_count, 2)
        self.assertIn("at most 2 output(s)", str(exc))

    def test_invalid_input_type_exception(self):
        # Operator has FloatTensorType input but we only allow Int64TensorType
        with self.assertRaises(InvalidInputTypeException) as cm:
            check_input_and_output_types(
                self.operator, good_input_types=[Int64TensorType]
            )

        exc = cm.exception
        self.assertEqual(exc.operator, self.operator)
        self.assertEqual(exc.variable, self.float_var1)
        self.assertEqual(exc.expected_types, [Int64TensorType])
        self.assertIn("wrong type", str(exc))
        self.assertIn("float_input_1", str(exc))

    def test_invalid_output_type_exception(self):
        # Operator has Int64TensorType output but we only allow FloatTensorType
        with self.assertRaises(InvalidOutputTypeException) as cm:
            check_input_and_output_types(
                self.operator, good_output_types=[FloatTensorType, StringTensorType]
            )

        exc = cm.exception
        self.assertEqual(exc.operator, self.operator)
        self.assertEqual(exc.variable, self.int_var1)
        self.assertEqual(exc.expected_types, [FloatTensorType, StringTensorType])
        self.assertIn("wrong type", str(exc))
        self.assertIn("int_input_1", str(exc))

    def test_valid_input_output_numbers(self):
        # This should not raise any exception
        try:
            check_input_and_output_numbers(
                self.operator, input_count_range=1, output_count_range=1
            )
        except Exception as e:
            self.fail(f"Valid input/output numbers raised exception: {e}")

    def test_valid_input_output_types(self):
        """Test that valid input/output types don't raise exceptions."""
        # This should not raise any exception
        try:
            check_input_and_output_types(
                self.operator,
                good_input_types=[FloatTensorType, Int64TensorType],
                good_output_types=[FloatTensorType, Int64TensorType],
            )
        except Exception as e:
            self.fail(f"Valid input/output types raised exception: {e}")

    def test_existing_invalid_input_length_exception_still_works(self):
        # Operator has 1 input but we require at least 2
        with self.assertRaises(InvalidInputLengthException) as cm:
            check_input_and_output_numbers(
                self.operator, input_count_range=[2, None], output_count_range=1
            )

        exc = cm.exception
        self.assertEqual(exc.operator, self.operator)
        self.assertEqual(exc.min_input_count, 2)
        self.assertIsNone(exc.max_input_count)
        self.assertIn("at least 2 input(s)", str(exc))

    def test_exception_hierarchy(self):
        # For backwards compatibility, ensure all our exceptions are RuntimeError subclasses
        self.assertTrue(issubclass(InvalidInputLengthException, RuntimeError))
        self.assertTrue(issubclass(InvalidOutputLengthException, RuntimeError))
        self.assertTrue(issubclass(InvalidInputTypeException, RuntimeError))
        self.assertTrue(issubclass(InvalidOutputTypeException, RuntimeError))

    def test_exception_attributes_accessible(self):
        try:
            check_input_and_output_numbers(self.operator, output_count_range=[2, 3])
        except InvalidOutputLengthException as e:
            # Test that we can access all expected attributes
            self.assertIsNotNone(e.operator)
            self.assertIsNotNone(e.min_output_count)
            self.assertIsNone(e.max_output_count)

        try:
            check_input_and_output_types(
                self.operator, good_input_types=[StringTensorType]
            )
        except InvalidInputTypeException as e:
            # Test that we can access all expected attributes
            self.assertIsNotNone(e.operator)
            self.assertIsNotNone(e.variable)
            self.assertIsNotNone(e.expected_types)

    def test_multiple_validation_errors(self):
        # Create an operator with multiple problems:
        #   wrong input count, wrong output count, wrong types
        bad_operator = self.scope.declare_local_operator("BadOperator")
        # No inputs (should need 1), No outputs (should need 2),
        #   wrong types don't matter here

        # Input count error should be caught first.
        # The retains the original error message for the validation errors.
        with self.assertRaises(InvalidInputLengthException):
            check_input_and_output_numbers(
                bad_operator, input_count_range=1, output_count_range=2
            )

    def test_exception_messages_contain_useful_info(self):
        try:
            check_input_and_output_numbers(self.operator, output_count_range=[2, 3])
        except InvalidOutputLengthException as e:
            msg = str(e)
            self.assertIn("LinearRegressor", msg)  # operator type
            self.assertIn("at least 2 output(s)", msg)  # expected count
            self.assertIn("we got 1 output(s)", msg)  # actual count
            self.assertIn(self.operator.full_name, msg)  # operator name

        try:
            check_input_and_output_types(
                self.operator, good_input_types=[StringTensorType]
            )
        except InvalidInputTypeException as e:
            msg = str(e)
            self.assertIn("LinearRegressor", msg)  # operator type
            self.assertIn("float_input_1", msg)  # variable name
            self.assertIn("wrong type", msg)  # error description
            self.assertIn("StringTensorType", msg)  # expected type


if __name__ == "__main__":
    unittest.main()
