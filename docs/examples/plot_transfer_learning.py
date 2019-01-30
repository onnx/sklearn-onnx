# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-transfer-learning:

Transfer learning with ONNX
===========================


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
# Transfer Learning with an image
# +++++++++++++++++++++++++++++++
#
# It starts by downloading a model from
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
plt.show()
