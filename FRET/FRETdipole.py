#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:53:56 2021

@author: albertsmith
"""

def bond(molsys,sel0,sel1,sel2,sel3):
    def rD():
        return 0.5*(sel0.positions+sel1.positions)
    def rA():
        return 0.5*(sel2.positions+sel3.positions)
    def vD():
        return (sel0.positions-sel1.positions)
    def vA():
        return (sel2.positions-sel3.positions)
    
    return rD,rA,vD,vA 