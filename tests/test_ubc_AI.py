#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pytest
import ubc_AI

testdir = os.path.dirname(os.path.abspath(__file__))
#print(testdir.split('/')[:-2]+['presto'])
#presto_dir = ('/').join(testdir.split('/')[:-2]+['presto'])
#print(presto_dir)
#sys.path.insert(0, presto_dir)
#import presto
from ubc_AI.data import pfdreader
#datadir = os.path.join(testdir, 'data')

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

	filename = args.filename
	bins = args.bins
	ndms = args.ndms
	"""
	apfd = pfdreader(filename)
	TvP = apfd.getdata(intervals=bins).reshape(bins,bins)
	FvP = apfd.getdata(subbands=bins).reshape(bins,bins)
	profile = apfd.getdata(phasebins=bins)
	DMc = apfd.getdata(DMbins=ndms)

	np.savetxt(filename+'.profile', profile)
	np.savetxt(filename+'.TvP', TvP)
	np.savetxt(filename+'.FvP', FvP)
	np.savetxt(filename+'.DMcurve', DMc)