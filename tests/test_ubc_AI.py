#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os

from ubc_AI.data import pfdreader

# import pickle
# import sys

# import presto


testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")


def test_pfdreader():
    """
    from ubc_AI.data import pfdreader
    import numpy as np
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='pfd to text')
    parser.add_argument('filename', default='', nargs='?', type=str, help='name of the input filterbank file')
    parser.add_argument('-b', dest='bins',  type=int, default=64, help='dimention of the feature (default 64)')
    parser.add_argument('-ndms', dest='ndms',  type=int, default=64, help='number of bins in the DM curve (default 64)')
    args = parser.parse_args()
    """
    filename = glob.glob(datadir + "/*.pfd")[0]
    bins = 64
    ndms = 64
    apfd = pfdreader(filename)
    apfd.getdata(intervals=bins).reshape(bins, bins)
    apfd.getdata(subbands=bins).reshape(bins, bins)
    apfd.getdata(phasebins=bins)
    apfd.getdata(DMbins=ndms)


"""
def test_quickclf():
    trained_ai_path = ('/').join(ubc_AI.__path__[0].split('/')[:-1])+'/data/trained_AI'
    print(trained_ai_path)

    with open(trained_ai_path+'/clfl2_PALFA.pkl','rb') as f:
        classifier = pickle.load(f)

    pfdfiles = glob.glob(datadir + '*.pfd')
    AI_scores = classifier.report_score([pfdreader(f) for f in pfdfile])
    for i,pfdfile in enumerate(pfdfiles):
        print(pfdfile,"Score:",AI_scores[i])
"""
