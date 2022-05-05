#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import pytest
import presto

import tensorflow as tf

import ubc_AI
from ubc_AI import TF_cnn

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")


def test_CNN():
    tCNN = TF_cnn.CNN()


# def test_metaCNN():
# 	tmCNN = TF_cnn.MetaCNN()
