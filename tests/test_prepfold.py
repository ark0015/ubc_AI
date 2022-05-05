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
    #pfd.combine_profs(new_npart, new_nsub)
    pfd.kill_intervals(self, intervals)
    pfd.kill_subbands(self, subbands)
    pfd.plot_sumprof(self, device="/xwin")
    pfd.greyscale(self, array2d, **kwargs)
    pfd.plot_intervals(self, phasebins="All", device="/xwin")
    pfd.plot_subbands(self, phasebins="All", device="/xwin")
    pfd.calc_varprof(self)
    pfd.calc_redchi2()
    pfd.plot_chi2_vs_DM()
    pfd.plot_chi2_vs_sub()
    pfd.estimate_offsignal_redchi2()
    pfd.adjust_fold_frequency()
    pfd.dynamic_spectra(onbins)

def test_pfddata(pfd_file):
    pfddata = prepfold.pfddata(pfd_file)
    pfddata.getdata()
