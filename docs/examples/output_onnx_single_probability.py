from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnx.helper as helper

# Convert a RandomForestClassifier() model to ONNX format such that it outputs a single probability as a float32.
# This code reshapes the output to ensure it is a single float32 value instead of a tuple of probabilities


# Configuration options to set the model to be converted and the output filename
# Set the model to be converted (e.g., RandomForest classifier)
model_to_convert = rfc  # Replace with the model you want to convert

# Set the output filename for the modified ONNX model
output_filename = "output_file.onnx"  # Replace with your desired output filename

# Step 1: Convert the model to ONNX format, disabling the output of labels.
# Define the input type for the ONNX model. The input type is a float tensor with shape
# [None, X_test.shape[1]], where None indicates that the number of input samples can be flexible,
# and X_test.shape[1] is the number of features for each input sample.
# A "tensor" is essentially a multi-dimensional array, commonly used in machine learning to represent data.
# A "float tensor" specifically contains floating-point numbers, which are numbers with decimals.
initial_type = [('float_input', FloatTensorType([None, X_test.shape[1]]))]

# Convert the model to ONNX format.
# - target_opset=12 specifies the version of ONNX operators to use.
# - options={...} sets parameters for the conversion:
#   - "zipmap": False ensures that the output is a raw array of probabilities instead of a dictionary.
#   - "output_class_labels": False ensures that the output contains only probabilities, not class labels.
# ONNX (Open Neural Network Exchange) is an open format for representing machine learning models.
# It allows interoperability between different machine learning frameworks, enabling the use of models across various platforms.
onx = convert_sklearn(
    model_to_convert,
    initial_types=initial_type,
    target_opset=12,
    options={id(model_to_convert): {"zipmap": False, "output_class_labels": False}}  # Ensures the output is only probabilities, not labels
)

# Step 2: Load the ONNX model for further modifications if needed
# Load the ONNX model from the serialized string representation.
# An ONNX file is essentially a serialized representation of a machine learning model that can be shared and used across different systems.
onnx_model = onnx.load_model_from_string(onx.SerializeToString())

# Assuming the first output in this model should be the probability tensor
# Extract the name of the output tensor representing the probabilities.
# If there are multiple outputs, select the second one, otherwise, select the first.
prob_output_name = onnx_model.graph.output[1].name if len(onnx_model.graph.output) > 1 else onnx_model.graph.output[0].name

# Add a Gather node to extract only the probability of the positive class (index 1)
# Create a tensor to specify the index to gather (index 1), which represents the positive class.
indices = helper.make_tensor("indices", onnx.TensorProto.INT64, (1,), [1])  # Index 1 to gather positive class

# Create a "Gather" node in the ONNX graph to extract the probability of the positive class.
# - inputs: [prob_output_name, "indices"] specify the inputs to this node (probability tensor and index tensor).
# - outputs: ["positive_class_prob"] specify the name of the output of this node.
# - axis=1 indicates gathering along the columns (features) of the probability tensor.
# A "Gather" node is used to extract specific elements from a tensor. Here, it extracts the probability for the positive class.
gather_node = helper.make_node(
    "Gather",
    inputs=[prob_output_name, "indices"],
    outputs=["positive_class_prob"],
    axis=1  # Gather along columns (axis 1)
)

# Add the Gather node to the ONNX graph
onnx_model.graph.node.append(gather_node)

# Add the tensor initializer for indices (needed for the Gather node)
# Initializers in ONNX are used to define constant tensors that are used in the computation.
onnx_model.graph.initializer.append(indices)

# Remove existing outputs and add only the new output for the positive class probability
# Clear the existing output definitions to replace them with the new output.
while len(onnx_model.graph.output) > 0:
    onnx_model.graph.output.pop()

# Define new output for the positive class probability
# Create a new output tensor specification with the name "positive_class_prob".
positive_class_output = helper.make_tensor_value_info("positive_class_prob", onnx.TensorProto.FLOAT, [None, 1])
onnx_model.graph.output.append(positive_class_output)

# Step 3: Save the modified ONNX model
# Save the modified ONNX model to the specified output filename.
# The resulting ONNX file can then be loaded and used in different environments that support ONNX, such as inference servers or other machine learning frameworks.
with open(output_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())
