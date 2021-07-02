# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages
import os
this = os.path.dirname(__file__)

with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]

packages = find_packages()
assert packages

# read version from the package file.
version_str = '1.0.0'
with (open(os.path.join(this, 'skl2onnx/__init__.py'), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

README = os.path.join(os.getcwd(), "README.md")
with open(README) as f:
    long_description = f.read()
    start_pos = long_description.find('## Introduction')
    if start_pos >= 0:
        long_description = long_description[start_pos:]

setup(
    name='skl2onnx',
    version=version_str,
    description="Convert scikit-learn models to ONNX",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache License v2.0',
    author='Microsoft Corporation',
    author_email='onnx@microsoft.com',
    url='https://github.com/onnx/sklearn-onnx',
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License'],
)
