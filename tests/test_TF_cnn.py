#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import pytest
import presto

import ubc_AI
from ubc_AI import TF_cnn

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")

def test_CNN():
	input_x = tf.variable("x")
	tCNN = TF_cnn.CNN(input_x)
#def test_metaCNN():
#	tCNN = TF_cnn.CNN()