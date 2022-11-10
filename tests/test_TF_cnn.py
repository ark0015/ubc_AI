#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pytest
import numpy as np

from ubc_AI import TF_cnn
from ubc_AI import classifier
from ubc_AI.data import dataloader


testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")


@pytest.fixture
def pfd_data():
    """
    load data
    Class dataloader() will load data from either a pickled dataloader object or a text file
    with a list of pfd files and their classifications. The classifications should follow the
    pfd file name(with path) and can be either in 1 column format or 5 column format. In 5
    column format, the classifications should be in the order:
    pulsar | pulse profile | DM curve | time-vs-phase | frequency-vs-phase
    """
    test_files = glob.glob(datadir + "/*.pfd")
    save_array = []
    for i, tf in enumerate(test_files):
        if i < 5:
            save_array.append([tf, 0, 0, 0, 0, 0])
        else:
            save_array.append([tf, 1, 1, 1, 1, 1])
    np.savetxt(datadir + "/Testing.txt", save_array, fmt="%s", delimiter=" ")
    return dataloader(datadir + "/Testing.txt")



def test_CNN(pfd_data):
    cnn = TF_cnn.CNN()
    # cnn.fit(pfd_data.pfds, pfd_data.target)


def test_pnnclf(pfd_data):
    pnn1 = classifier.pnnclf(
        design=[25], gamma=0.5, feature={"phasebins": 64}, maxiter=None
    )  # F1 .74
    pnn1.fit(pfd_data.pfds, pfd_data.target)

    pnn2 = classifier.pnnclf(
        design=[9],
        gamma=0.001,
        use_pca=True,
        n_comp=24,
        feature={"intervals": 64},
        maxiter=None,
    )  # F1 .83
    pnn2.fit(pfd_data.pfds, pfd_data.target)

    pnn3 = classifier.pnnclf(
        design=[9], gamma=0.1, feature={"DMbins": 60}, maxiter=None
    )  # F1 .80
    pnn3.fit(pfd_data.pfds, pfd_data.target)


def test_cnnclf(pfd_data):
    cnn1 = classifier.cnnclf(
        feature={"intervals": 48},
        poolsize=[(3, 3), (2, 2)],
        n_epochs=65,
        batch_size=20,
        nkerns=[20, 40],
        filters=[16, 8],
        L1_reg=1.0,
        L2_reg=1.0,
    )
    cnn1.fit(pfd_data.pfds, pfd_data.target)
    #print(cnn1.model.summary())
    #print("1 Done.")
    cnn2 = classifier.cnnclf(
        feature={"subbands": 48},
        poolsize=[(3, 3), (2, 2)],
        n_epochs=65,
        batch_size=20,
        nkerns=[20, 40],
        filters=[16, 8],
        L1_reg=1.0,
        L2_reg=1.0,
    )
    cnn2.fit(pfd_data.pfds, pfd_data.target)
    #print("2 Done.")
    #print(cnn1.model.summary())


def test_svmclf(pfd_data):
    svmclf1 = classifier.svmclf(gamma=0.05, C=1.0, feature={"phasebins": 64}, probability=True)
    classifier.svmclf(
        gamma=0.005,
        C=5,
        feature={"intervals": 64},
        use_pca=True,
        n_comp=24,
        probability=True,
    )
    classifier.svmclf(
        gamma=0.001,
        C=24.0,
        feature={"subbands": 64},
        use_pca=True,
        n_comp=24,
        probability=True,
    )
    classifier.svmclf(gamma=0.2, C=25.0, feature={"DMbins": 60}, probability=True)
    classifier.svmclf(gamma=85.0, C=1.0, feature={"timebins": 52})


def test_LRclf(pfd_data):
    classifier.LRclf(
        C=32.0, penalty="l2", use_pca=True, n_comp=32, feature={"subbands": 52}
    )  # F1 .79
    classifier.LRclf(
        C=50.0, penalty="l2", use_pca=True, n_comp=24, feature={"intervals": 64}
    )  # F1 .80
    classifier.LRclf(C=0.07, penalty="l2", feature={"DMbins": 60})


def test_dtreeclf():
    tree3 = classifier.dtreeclf(
        min_samples_split=8, min_samples_leaf=4, feature={"DMbins": 60}
    )


def test_combined_AI(pfd_data):
    nn1 = classifier.pnnclf(
        design=[25], gamma=0.5, feature={"phasebins": 64}, maxiter=None
    )  # F1 .74
    nn2 = classifier.cnnclf(
        feature={"intervals": 48},
        poolsize=[(3, 3), (2, 2)],
        n_epochs=65,
        batch_size=20,
        nkerns=[20, 40],
        filters=[16, 8],
        L1_reg=1.0,
        L2_reg=1.0,
    )
    nn3 = classifier.pnnclf(
        design=[9],
        gamma=0.001,
        use_pca=True,
        n_comp=24,
        feature={"timebins": 64},
        maxiter=None,
    )  # F1 .83
    nn4 = classifier.cnnclf(
        feature={"subbands": 48},
        poolsize=[(3, 3), (2, 2)],
        n_epochs=65,
        batch_size=20,
        nkerns=[20, 40],
        filters=[16, 8],
        L1_reg=1.0,
        L2_reg=1.0,
    )
    nn5 = classifier.pnnclf(
        design=[9], gamma=0.1, feature={"DMbins": 60}, maxiter=None
    )  # F1 .80
    clf1 = classifier.svmclf(
        gamma=0.05, C=1.0, feature={"phasebins": 64}, probability=True
    )
    clf2 = classifier.svmclf(
        gamma=0.005,
        C=5,
        feature={"intervals": 64},
        use_pca=True,
        n_comp=18,
        probability=True,
    )
    clf3 = classifier.svmclf(
        gamma=0.001,
        C=24.0,
        feature={"subbands": 64},
        use_pca=True,
        n_comp=17,
        probability=True,
    )
    clf4 = classifier.svmclf(
        gamma=0.2, C=25.0, feature={"DMbins": 60}, probability=True
    )
    clf5 = classifier.svmclf(gamma=85.0, C=1.0, feature={"timebins": 52})
    lg1 = classifier.LRclf(
        C=32.0, penalty="l2", use_pca=True, n_comp=32, feature={"subbands": 52}
    )  # F1 .79
    lg2 = classifier.LRclf(
        C=50.0, penalty="l2", use_pca=True, n_comp=24, feature={"intervals": 64}
    )  # F1 .80
    lg3 = classifier.LRclf(C=0.07, penalty="l2", feature={"DMbins": 60})
    tree3 = classifier.dtreeclf(
        min_samples_split=8, min_samples_leaf=4, feature={"DMbins": 60}
    )
    # join the layer-1 classifiers into a combined-AI
    AIs = [nn1, nn2, nn3, nn4, nn5, clf1, clf2, clf3, clf4, clf5, tree3, lg1, lg2]
    # good combos discovered so far:
    combo = [0, 1, 3, 4, 5, 6, 7, 8]  # F1 = 0.9
    # combo = [ 5, 1, 3, 8] #F1 = 0.9
    # combo = [0, 1, 3, 4] #F1 = 0.9
    # combo = [5, 6, 7, 8] #F1 = 0.9

    clfl2 = classifier.combinedAI(
        [AIs[i] for i in combo], strategy="lr", C=0.1, penalty="l2"
    )

    # training the combined AI
    clfl2.fit(pfd_data.pfds, pfd_data.target)


os.remove(datadir + "/Testing.txt")
