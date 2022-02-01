#!/usr/bin/env python

from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 7, \
    "SPACE is designed to work with Python 3.7 and greater. " \
    + "Please install it before proceeding."

setup(
    name='space',
    packages=['space'],
    version='1.1.0',
    install_requires=[
        'attrdict~=2.0.1',
        'yacs~=0.1.8',
        'torch~=1.10.2'
    ],
)
