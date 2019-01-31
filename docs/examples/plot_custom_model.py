# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-custom-model:

Write your own converter for your own model
===========================================

It might happens that you implemented your own model
and there is obviously no existing converter for this
new model. That does not mean the conversion of a pipeline
which includes it would not work. Let's see how to do it.

`t-SNE <https://lvdmaaten.github.io/tsne/>`_ is an interesting 
transform which can only be used to study data as there is no
way to reproduce the result once it was fitted. That's why
the class `TSNE <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`_
does not have any method *transform*, only
`fit_transform <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE.fit_transform>`_.
This example proposes a way to train a machine learned model
which approximates the outputs of a *t-SNE* transformer.


.. contents::
    :local:

Implementation of the new transform
+++++++++++++++++++++++++++++++++++

"""
