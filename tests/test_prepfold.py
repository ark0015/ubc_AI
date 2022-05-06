#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
import glob

import ubc_AI
from ubc_AI import prepfold

@pytest.fixture
def pfd_file():
    testdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(testdir, "data")
    pfd_file = glob.glob(datadir + "/*.pfd")[0]
    return pfd_file

def test_pfd(pfd_file):
    pfd = prepfold.pfd(pfd_file)
    pfd.dedisperse()
    pfd.freq_offsets()
    pfd.DOF_corr()
    pfd.use_for_timing()
    pfd.time_vs_phase()
    pfd.adjust_period()
    ##pfd.combine_profs(new_npart, new_nsub)
    ##pfd.kill_intervals(intervals)
    ##pfd.kill_subbands(subbands)
    ##pfd.greyscale(array2d)
    pfd.calc_varprof()
    pfd.calc_redchi2()
    pfd.estimate_offsignal_redchi2()
    #pfd.adjust_fold_frequency(phasebins)
    ##pfd.dynamic_spectra(onbins)

def test_pfd_plotting(pfd_file):
    pfd = prepfold.pfd(pfd_file)
    pfd.plot_sumprof()
    #pfd.plot_chi2_vs_DM(loDM, hiDM)
    #pfd.plot_chi2_vs_sub()
    #pfd.plot_intervals()
    #pfd.plot_subbands()

def test_pfddata(pfd_file):
    pfddata = prepfold.pfddata(pfd_file)
    pfddata.getdata()
