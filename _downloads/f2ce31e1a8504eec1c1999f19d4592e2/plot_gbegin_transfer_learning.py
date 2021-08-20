# SPDX-License-Identifier: Apache-2.0

"""
Transfer Learning with ONNX
===========================

.. index:: transfer learning, deep learning

Transfer learning is common with deep learning.
A deep learning model is used as preprocessing before
the output is sent to a final classifier or regressor.
It is not quite easy in this case to mix framework,
:epkg:`scikit-learn` with :epkg:`pytorch`
(or :epkg:`skorch`), the Keras API for Tensorflow,
`tf.keras.wrappers.scikit_learn
<https://www.tensorflow.org/api_docs/python/tf/
keras/wrappers/scikit_learn>`_. Every combination
requires work. ONNX reduces the number of platforms to
support. Once the model is converted into ONNX,
it can be inserted in any :epkg:`scikit-learn` pipeline.

.. contents::
    :local:

Retrieve and load a model
+++++++++++++++++++++++++

We download one model from the :epkg:`ONNX Zoo` but the model
could be trained and produced by another converter library.
"""
import sys
from io import BytesIO
import onnx
from mlprodict.sklapi import OnnxTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from mlinsights.plotting.gallery import plot_gallery_images
import matplotlib.pyplot as plt
from skl2onnx.tutorial.imagenet_classes import class_names
import numpy
from PIL import Image
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
import os
import urllib.request


def download_file(url, name, min_size):
    if not os.path.exists(name):
        print("download '%s'" % url)
        with urllib.request.urlopen(url) as u:
            content = u.read()
        if len(content) < min_size:
            raise RuntimeError(
                "Unable to download '{}' due to\n{}".format(
                    url, content))
        print("downloaded %d bytes." % len(content))
        with open(name, "wb") as f:
            f.write(content)
    else:
        print("'%s' already downloaded" % name)


model_name = "squeezenet1.1-7.onnx"
url_name = ("https://github.com/onnx/models/raw/master/vision/"
            "classification/squeezenet/model")
url_name += "/" + model_name
try:
    download_file(url_name, model_name, 100000)
except RuntimeError as e:
    print(e)
    sys.exit(1)


################################################
# Loading the ONNX file and use it on one image.

sess = InferenceSession(model_name)

for inp in sess.get_inputs():
    print(inp)

#####################################
# The model expects a series of images of size
# `[3, 224, 224]`.

##########################################
# Classifying an image
# ++++++++++++++++++++

url = ("https://upload.wikimedia.org/wikipedia/commons/d/d2/"
       "East_Coker_elm%2C_2.jpg")
img = "East_Coker_elm.jpg"
download_file(url, img, 100000)

im0 = Image.open(img)
im = im0.resize((224, 224))
# im.show()

######################################
# Image to numpy and predection.


def im2array(im):
    X = numpy.asarray(im)
    X = X.transpose(2, 0, 1)
    X = X.reshape(1, 3, 224, 224)
    return X


X = im2array(im)
out = sess.run(None, {'data': X.astype(numpy.float32)})
out = out[0]

print(out[0, :5])

#####################################
# Interpretation


res = list(sorted((r, class_names[i]) for i, r in enumerate(out[0])))
print(res[-5:])

##########################################
# Classifying more images
# +++++++++++++++++++++++
#
# The initial image is rotated,
# the answer is changing.

angles = [a * 2. for a in range(-6, 6)]
imgs = [(angle, im0.rotate(angle).resize((224, 224)))
        for angle in angles]


def classify(imgs):
    labels = []
    for angle, img in imgs:
        X = im2array(img)
        probs = sess.run(None, {'data': X.astype(numpy.float32)})[0]
        pl = list(sorted(
            ((r, class_names[i]) for i, r in enumerate(probs[0])),
            reverse=True))
        labels.append((angle, pl))
    return labels


climgs = classify(imgs)
for angle, res in climgs:
    print("angle={} - {}".format(angle, res[:5]))


plot_gallery_images([img[1] for img in imgs],
                    [img[1][0][1][:15] for img in climgs])

#########################################
# Transfer learning in a pipeline
# +++++++++++++++++++++++++++++++
#
# The proposed transfer learning consists
# using a PCA to projet the probabilities
# on a graph.


with open(model_name, 'rb') as f:
    model_bytes = f.read()

pipe = Pipeline(steps=[
    ('deep', OnnxTransformer(
        model_bytes, runtime='onnxruntime1', change_batch_size=0)),
    ('pca', PCA(2))
])

X_train = numpy.vstack(
    [im2array(img) for _, img in imgs]).astype(numpy.float32)
pipe.fit(X_train)

proj = pipe.transform(X_train)
print(proj)

###########################################
# Graph for the PCA
# -----------------

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(proj[:, 0], proj[:, 1], 'o')
ax.set_title("Projection of classification probabilities")
text = ["%1.0f-%s" % (el[0], el[1][0][1]) for el in climgs]
for label, x, y in zip(text, proj[:, 0], proj[:, 1]):
    ax.annotate(
        label, xy=(x, y), xytext=(-10, 10), fontsize=8,
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

###########################################
# Remove one layer at the end
# ---------------------------
#
# The last is often removed before the model is
# inserted in a pipeline. Let's see how to do that.
# First, we need the list of output for every node.


model_onnx = onnx.load(BytesIO(model_bytes))
outputs = []
for node in model_onnx.graph.node:
    print(node.name, node.output)
    outputs.extend(node.output)

#################################
# We select one of the last one.

selected = outputs[-3]
print("selected", selected)

#################################
# And we tell *OnnxTransformer* to use that
# specific one and to flatten the output
# as the dimension is not a matrix.


pipe2 = Pipeline(steps=[
    ('deep', OnnxTransformer(
        model_bytes, runtime='onnxruntime1', change_batch_size=0,
        output_name=selected, reshape=True)),
    ('pca', PCA(2))
])

try:
    pipe2.fit(X_train)
except InvalidArgument as e:
    print("Unable to fit due to", e)

#######################################
# We check that it is different.
# The following values are the shape of the
# PCA components. The number of column is the number
# of dimensions of the outputs of the transfered
# neural network.

print(pipe.steps[1][1].components_.shape,
      pipe2.steps[1][1].components_.shape)

#######################################
# Graph again.

proj2 = pipe2.transform(X_train)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(proj2[:, 0], proj2[:, 1], 'o')
ax.set_title("Second projection of classification probabilities")
text = ["%1.0f-%s" % (el[0], el[1][0][1]) for el in climgs]
for label, x, y in zip(text, proj2[:, 0], proj2[:, 1]):
    ax.annotate(
        label, xy=(x, y), xytext=(-10, 10), fontsize=8,
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
