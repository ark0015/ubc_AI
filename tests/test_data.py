#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np

from ubc_AI import TF_cnn
from ubc_AI.data import dataloader


testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")


def test_CNN():
    test_files = glob.glob(datadir + "/*.pfd")
    save_array = []
    for tf in test_files:
        save_array.append([tf, 1, 1, 1, 1, 1])
    np.savetxt(datadir + "/Testing.txt", save_array, fmt="%s", delimiter=" ")
    loaded_data_files = dataloader(datadir + "/Testing.txt")
    loaded_data_files.pfds
    loaded_data_files.target
    os.remove(datadir + "/Testing.txt")


# def test_metaCNN():
#   tmCNN = TF_cnn.MetaCNN()
