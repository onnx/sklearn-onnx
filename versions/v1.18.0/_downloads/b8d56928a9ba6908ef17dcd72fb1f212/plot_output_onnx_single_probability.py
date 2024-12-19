"""
Append onnx nodes to the converted model
========================================

This example show how to append some onnx nodes to the converted
model to produce the desired output. In this case, it removes the second
column of the output probabilies.

To be completly accurate, most of the code was generated using a LLM
and modified to accomodate with the latest changes.
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression(max_iter=500)
clr.fit(X_train, y_train)


################################################
# model_to_convert refers to the scikit-learn classifier to convert.
model_to_convert = clr  # model to convert
X_test = X_test[:1]  # data used to test or train, one row is enough

################################################
# Set the output filename for the modified ONNX model
output_filename = "output_file.onnx"  # Replace with your desired output filename

################################################
# Step 1: Convert the model to ONNX format,
# disabling the output of labels.
# Define the input type for the ONNX model.
# The input type is a float tensor with shape
# [None, X_test.shape[1]], where None indicates that the
# number of input samples can be flexible,
# and X_test.shape[1] is the number of features for each input sample.
# A "tensor" is essentially a multi-dimensional array,
# commonly used in machine learning to represent data.
# A "float tensor" specifically contains floating-point
# numbers, which are numbers with decimals.
initial_type = [("float_input", FloatTensorType([None, X_test.shape[1]]))]

################################################
# Convert the model to ONNX format.
# - target_opset=18 specifies the version of ONNX operators to use.
# - options={...} sets parameters for the conversion:
#   - "zipmap": False ensures that the output is a raw array
#   - of probabilities instead of a dictionary.
#   - "output_class_labels": False ensures that the output
#     contains only probabilities, not class labels.
# ONNX (Open Neural Network Exchange) is an open format for
# representing machine learning models.
# It allows interoperability between different machine learning frameworks,
# enabling the use of models across various platforms.
onx = convert_sklearn(
    model_to_convert,
    initial_types=initial_type,
    target_opset={"": 18, "ai.onnx.ml": 3},
    options={
        id(model_to_convert): {"zipmap": False, "output_class_labels": False}
    },  # Ensures the output is only probabilities, not labels
)

################################################
# Step 2: Load the ONNX model for further modifications if needed
# Load the ONNX model from the serialized string representation.
# An ONNX file is essentially a serialized representation of a machine learning
# model that can be shared and used across different systems.
onnx_model = onnx.load_model_from_string(onx.SerializeToString())

################################################
# Assuming the first output in this model should be the probability tensor
# Extract the name of the output tensor representing the probabilities.
# If there are multiple outputs, select the second one, otherwise, select the first.
prob_output_name = (
    onnx_model.graph.output[1].name
    if len(onnx_model.graph.output) > 1
    else onnx_model.graph.output[0].name
)

################################################
# Add a Gather node to extract only the probability
# of the positive class (index 1)
# Create a tensor to specify the index to gather
# (index 1), which represents the positive class.
indices = onnx.helper.make_tensor(
    "indices", onnx.TensorProto.INT64, (1,), [1]
)  # Index 1 to gather positive class

################################################
# Create a "Gather" node in the ONNX graph to extract the probability of the positive class.
# - inputs: [prob_output_name, "indices"] specify the inputs
#   to this node (probability tensor and index tensor).
# - outputs: ["positive_class_prob"] specify the name of the output of this node.
# - axis=1 indicates gathering along the columns (features) of the probability tensor.
# A "Gather" node is used to extract specific elements from a tensor.
# Here, it extracts the probability for the positive class.
gather_node = onnx.helper.make_node(
    "Gather",
    inputs=[prob_output_name, "indices"],
    outputs=["positive_class_prob"],
    axis=1,  # Gather along columns (axis 1)
)

################################################
# Add the Gather node to the ONNX graph
onnx_model.graph.node.append(gather_node)

################################################
# Add the tensor initializer for indices (needed for the Gather node)
# Initializers in ONNX are used to define constant tensors that are used in the computation.
onnx_model.graph.initializer.append(indices)

################################################
# Remove existing outputs and add only the new output for the positive class probability
# Clear the existing output definitions to replace them with the new output.
del onnx_model.graph.output[:]

################################################
# Define new output for the positive class probability
# Create a new output tensor specification with the name "positive_class_prob".
positive_class_output = onnx.helper.make_tensor_value_info(
    "positive_class_prob", onnx.TensorProto.FLOAT, [None, 1]
)
onnx_model.graph.output.append(positive_class_output)

################################################
# Step 3: Save the modified ONNX model
# Save the modified ONNX model to the specified output filename.
# The resulting ONNX file can then be loaded and used in different environments
# that support ONNX, such as inference servers or other machine learning frameworks.
onnx.save(onnx_model, output_filename)


################################################
# The model can be printed as follows.
print(onnx.printer.to_text(onnx_model))
