#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:53:56 2021

@author: albertsmith
"""

def bond(molsys,sel1,sel2,sel3,sel4):
    def rD():
        return 0.5*(sel1.positions+sel2.positions)
    def rA():
        return 0.5*(sel3.positions+sel4.positions)
    def vD():
        return (sel1.positions-sel2.positions)
    def vA():
        return (sel3.positions-sel4.positions)
    
    return rD,rA,vD,vA 