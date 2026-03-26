#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:53:56 2021

@author: albertsmith
"""


from ..Selection.MolSys import MolSelect


def bond(molsys,sel1,sel2,sel3,sel4):
    def rD():
        return 0.5*(sel1.positions+sel2.positions)
    def rA():
        return 0.5*(sel3.positions+sel4.positions)
    def vD():
        return (sel1.positions-sel2.positions)
    def vA():
        return (sel3.positions-sel4.positions)
    
    select=MolSelect(molsys=molsys)
    select.sel1=[s1+s2 for s1,s2 in zip(sel1,sel2)]
    select.sel1=[s1+s2 for s1,s2 in zip(sel3,sel4)]
    select.repr_sel=[s1+s2+s3+s4 for s1,s2,s3,s4 in zip(sel1,sel2,sel3,sel4)]
    
    
    out={'rD':rD,'rA':rA,'vD':vD,'vA':vA,'select':select}
    
    return out 