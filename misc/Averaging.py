#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:37:35 2022

@author: albertsmith
"""

from pyDR import clsDict

def avg2sel(data,sel):
    """
    Averages data points in a data object together such that we may define a new
    data object whose selection matches a given input selection.

    Parameters
    ----------
    data : pyDR.Data
        Data object to be averaged. Must have its selection specified. We assume
        that this selection consists of single bonds (usually we're averaging 
        an MD trajectory, which does not have ambiguity in its assignment)
    
    sel : pyDR.MolSelect
        Selection specifying how to average the initial data object. Note that
        each selection in sel defines one or more bonds. If one of the bonds
        is not existing in the data object's selection, then that bond will be
        skipped entirely. Note that each group selection in sel is expected to
        have equal number of atoms in both sel1 and sel2, corresponding to
        each bond. If this is not the case, then the heavy atom will be ignored
        and we will just try to match the light atom (proton) in data.
    
    Returns
    -------
    pyDR.Data
    Averaged data object.

    """
    
    detect=data.detect.copy()
    out=clsDict['Data'](sens=detect,src_data=data) #Create output data with sensitivity as input detectors
    out.label=data.label


    sel1=list()
    sel2=list()
    
    flds=['R','Rstd','S2','S2std']
    
    for s1,s2 in zip(sel.sel1,sel.sel2):
        