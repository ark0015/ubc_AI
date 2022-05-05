#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

import ubc_AI
from ubc_AI import known_pulsars


def test_ATNF_pulsarlist():
    known_pulsars.ATNF_pulsarlist()


def test_GBNCC_pulsarlist():
    known_pulsars.GBNCC_pulsarlist()


def test_LOFAR_pulsarlist():
    known_pulsars.LOFAR_pulsarlist()


def test_PALFA_pulsarlist():
    known_pulsars.PALFA_pulsarlist()


def test_PALFA_jodrell_extrainfo():
    known_pulsars.PALFA_jodrell_extrainfo()


def test_ao327_pulsarlist():
    known_pulsars.ao327_pulsarlist()


def test_FERMI_pulsarlist():
    known_pulsars.FERMI_pulsarlist()


def test_GBT350NGP_pulsarlist():
    known_pulsars.GBT350NGP_pulsarlist()


def test_ryan_pulsarlist():
    known_pulsars.ryan_pulsars()


def test_get_allpulsars():
    known_pulsars.get_allpulsars()
