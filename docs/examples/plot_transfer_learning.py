# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-transfer-learning:

Transfer learning with ONNX
===========================

`Transfer learning <https://en.wikipedia.org/wiki/Transfer_learning>`_
is usually useful to adapt a deep learning model to some
new problem for which the number of images is not enough
to train a deep learning model. The proposed solution
implies the use of class *OnnxTransformer* which wraps
*OnnxRuntime* into a *scikit-learn* transformer
easily pluggable into a pipeline.

.. contents::
    :local:

Train a model
+++++++++++++

A very basic example using random forest and
the iris dataset.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier(n_estimators=1, max_depth=2)
clr.fit(X_train, y_train)
print(clr)

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([1, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)

with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

###################################
# Compute ONNX prediction similarly as scikit-learn transformer
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from onnxruntime.sklapi import OnnxTransformer

with open("rf_iris.onnx", "rb") as f:
    content = f.read()

ot = OnnxTransformer(content, output_name="output_probability")
ot.fit(X_train, y_train)

print(ot.transform(X_test[:5]))

###################################
# .. index:: transfer learning, MobileNet, ImageNet
#
# Transfer Learning with MobileNet
# ++++++++++++++++++++++++++++++++
#
# Deep learning models started to win
# the `ImageNet <http://www.image-net.org/>`_
# competition in 2012 and most the winners
# are available on the web as pre-trained models.
# Transfer Learning is computed by wrapping
# a backend into a *scikit-learn*
# transformers which *onnxruntime* does.

import os
model_file = "mobilenetv2-1.0.onnx"
if not os.path.exists(model_file):
    print("Download '{0}'...".format(model_file))
    import urllib.request
    url = "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx"
    urllib.request.urlretrieve(url, model_file)
    print("Done.")

class_names = "imagenet_class_index.json"
if not os.path.exists(class_names):
    print("Download '{0}'...".format(class_names))
    import urllib.request
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    urllib.request.urlretrieve(url, class_names)
    print("Done.")

import json
with open(class_names, "r", encoding="utf-8") as f:
    content_classes = f.read()
labels = json.loads(content_classes)

#####################################
# Let's consider one image form *wikipedia*.

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy

img = Image.open('daisy_wikipedia.jpg')
plt.imshow(img)
plt.axis('off')

#####################################
# Let's create the OnnxTransformer
# which we apply on that particular image.

with open(model_file, "rb") as f:
    model_bytes = f.read()
    
ot = OnnxTransformer(model_bytes)

img2 = img.resize((224, 224))
X = numpy.asarray(img2).transpose((2, 0, 1))
X = X[numpy.newaxis, :, :, :] / 255.0
print(X.shape, X.min(), X.max())

pred = ot.fit_transform(X)[0, :]
print(pred.shape)

#############################
# And the best classes are...

from heapq import nlargest
results = nlargest(10, range(pred.shape[0]), pred.take)
print(results)

import pandas
data=[{"index": i, "label": labels.get(str(i), ('?', '?'))[1], 'score': pred[i]} \
      for i in results]
df = pandas.DataFrame(data)
print(df)

###################################
# .. index:: Yolo
# 
# Transfer Learning with Yolo
# +++++++++++++++++++++++++++
#
# `yolo <https://pjreddie.com/darknet/yolo/>`_
# is quite popular among the framework
# which can identity objects in images in real time.
# One of the models is available in 
# `ONNX zoo <https://github.com/onnx/models>`_.

import os
filename = "tiny_yolov2.tar.gz"
if not os.path.exists(filename):
    print("Download '{0}'...".format(filename))
    import urllib.request
    url = "https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz"
    urllib.request.urlretrieve(url, filename)
    print("Done.")

model_file = "tiny_yolov2/model.onnx"
if not os.path.exists(model_file):
    print("Unzip '{0}'.".format(model_file))
    import tarfile
    tfile = tarfile.open(filename, 'r:gz')
    tfile.extractall()
    print("Done.")

#######################################"
# Let's retrieve an image.

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy

img = Image.open('Au-Salon-de-l-agriculture-la-campagne-recrute.jpg')
plt.imshow(img)
plt.axis('off')

#################################
# It needs to be zoomed, converted into an array,
# resized, transposed.
img2 = img.resize((416, 416))
X = numpy.asarray(img2)
X = X.transpose(2,0,1)
X = X.reshape(1,3,416,416)
X = X.astype(numpy.float32)

#####################################
# Let's create the OnnxTransformer

with open(model_file, "rb") as f:
    model_bytes = f.read()
    
ot = OnnxTransformer(model_bytes)
pred = ot.fit_transform(X)
print(pred.shape)

######################################
# Let's display the results on the image itself

def display_yolo(img, out, threshold):
    """
    Displays yolo results *out* on an image *img*.
    *threshold* filters out uncertain results.
    """
    import numpy as np
    numClasses = 20
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    def sigmoid(x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def softmax(x):
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)

    clut = [(0,0,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,128),
            (128,255,0),(128,128,0),(0,128,255),(128,0,128),
            (255,0,128),(128,0,255),(255,128,128),(128,255,128),(255,255,0),
            (255,128,128),(128,128,255),(255,128,128),(128,255,128),(128,255,128)]
    label = ["aeroplane","bicycle","bird","boat","bottle",
             "bus","car","cat","chair","cow","diningtable",
             "dog","horse","motorbike","person","pottedplant",
             "sheep","sofa","train","tvmonitor"]

    draw = ImageDraw.Draw(img)
    for cy in range(0,13):
        for cx in range(0,13):
            for b in range(0,5):
                channel = b*(numClasses+5)
                tx = out[channel  ][cy][cx]
                ty = out[channel+1][cy][cx]
                tw = out[channel+2][cy][cx]
                th = out[channel+3][cy][cx]
                tc = out[channel+4][cy][cx]

                x = (float(cx) + sigmoid(tx))*32
                y = (float(cy) + sigmoid(ty))*32

                w = np.exp(tw) * 32 * anchors[2*b  ]
                h = np.exp(th) * 32 * anchors[2*b+1]

                confidence = sigmoid(tc)

                classes = np.zeros(numClasses)
                for c in range(0, numClasses):
                    classes[c] = out[channel + 5 +c][cy][cx]
                    classes = softmax(classes)
                detectedClass = classes.argmax()

                if threshold < classes[detectedClass] * confidence:
                    color = clut[detectedClass]
                    x = x - w/2
                    y = y - h/2
                    draw.line((x  ,y  ,x+w,y ),fill=color, width=3)
                    draw.line((x  ,y  ,x  ,y+h),fill=color, width=3)
                    draw.line((x+w,y  ,x+w,y+h),fill=color, width=3)
                    draw.line((x  ,y+h,x+w,y+h),fill=color, width=3)

    return img

img_results = display_yolo(img2, pred[0], 0.038)
plt.imshow(img_results)
plt.axis('off')

#################################
# **Versions used for this example**

import numpy, sklearn
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
import onnx, onnxruntime, skl2onnx, onnxmltools, lightgbm
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("skl2onnx: ", skl2onnx.__version__)
