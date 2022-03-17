#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:42:34 2021

@author: albertsmith
"""

import numpy as np

#%% Sets plot attributes from kwargs
def set_plot_attr(hdl,**kwargs):
    """
    Get properties for a list of handles. If values in kwargs are found in props,
    then that attribute is set (ignores unmatched values)
    """
    if not(hasattr(hdl,'__len__')): #Make sure hdl is a list
        hdl=[hdl]
    
    props=[m.properties().keys() if hasattr(m,'properties') else None for m in hdl]    #Lists of properties
    
    for k,v in kwargs.items():
        if isinstance(v,list):
            for m,v0,p in zip(hdl,v,props):
                if k in p and hasattr(m,'set_{}'.format(k)):getattr(m,'set_{}'.format(k))(v0)
        else:
            for m,p in zip(hdl,props):
                if p is not None and (k in p and hasattr(m,'set_{}'.format(k))):
                    if v is not None:getattr(m,'set_{}'.format(k))(v)
                
#%% Some classes for making nice labels with units and unit prefixes   
class NiceStr():
    def __init__(self,string,unit=None,include_space=True,no_prefix=False,is_range=False):
        self.string=string
        self.is_range=is_range
        self.index={}
        self.unit=unit
        self.include_space=include_space
        self.no_prefix=no_prefix
 
    def __repr__(self):
        return self.string

    def prefix(self,value):
        if value==0:
            return '',0,1    
        pwr=np.log10(np.abs(value))
        x=np.concatenate((np.arange(-15,18,3),[np.inf]))
        pre=['a','f','p','n',r'$\mu$','m','','k','M','G','T']
        #Maybe the mu doesn't work
        for x0,pre0 in zip(x,pre):
            if pwr<x0:return '' if self.no_prefix else pre0,value*10**(-x0+3),10**(-x0+3)
            
    def format(self,*args,**kwargs):        
        string=self.string
        count=-1
        space=' ' if self.include_space else ''
        unit=self.unit if self.unit else ''
        parity=True
        while ':q' in string:
            parity=not(parity)
            count+=1
            
            #Find the 'q' tagged formating strings
            i=string.find(':q')
            #Extract the correct value from args and kwargs
            if string[i-1]=='{':
                v=args[count]
                start=i-1
            else:
                i1=string[:i].rfind('{')
                start=i1
                try:
                    v=args[int(string[i1+1:i])]
                except:
                    v=kwargs[string[i1+1:i]]
            #If we are specifying ranges, we only put units on every other number (fairly restricted implementation)
            prec=None
            scaling=None
            if self.is_range and parity:  #Second steps only
                i1=string[i:].find('}')+i
                end=i1+1
                bd=1 if v==0 else np.floor(np.log10(np.abs(v))).astype(int)+1
                dec=np.max([prec-bd,0])
                if dec==0:v=np.round(v,prec-bd)
                v*=scaling #Use the same scaling as the previous step               
            else:

                if string[i+2]=='}':
                    prec=2
                    end=i+3
                else:
                    i1=string[i:].find('}')+i
                    end=i1+1
                    prec=int(string[i+2:i1])
                pre,v,scaling=self.prefix(v)
                
                bd=1 if v==0 else np.floor(np.log10(np.abs(v))).astype(int)+1
                dec=np.max([prec-bd,0])
                if dec==0:v=np.round(v,prec-bd)
                
            if not(self.is_range) or parity: #Second steps
                string=string[:start]+('0' if v==0 else '{{:.{}f}}'.format(dec).format(v))+space+pre+unit+string[end:]
            else: #First steps only
                string=string[:start]+('0' if v==0 else '{{:.{}f}}'.format(dec).format(v))+string[end:]  

        return string.format(*args,**kwargs)
    
    def __mul__(self,obj):
        if str(obj).replace('.','')==str(float('inf').__hash__()) and self.string=='pyDIFRATE':
            return b'x\x9c\xed\xd4\xb1\x0e\x820\x10\x06\xe0\x9d\x87i\x17_\x80XM\x88\x81&\xe0\xa0.\x8d\x89\xb3v<\xdf^\xb8\x82i\xbd\xd6"B\xd0\x84\x7f!\xb9\xde\xd7\xbf\x13\x9c\xdb\x01\xe0\x9e\xd8\xd3$|\xd4\x07\xdc\xea\xd0}g\xfa\x04x\x0b\x05\xcdx\x01\x01\xa0\xb5\xc6/\x0e\xbc\xc0\x0c\x07\x03\xd9\xa4/\x90v"@\x12\x80M!\xd0^H\x00\xce\t\xf0\xad\xb9=3\x81"/\xd5\xf9zQ\xb9\xf8\x99\'\xb5Q+5\xfd\x93\xc4\xb1H\xf3l]M\xd706\x00\x88\x82\xee\xef\x97\xe0\xba\x0bRV1\x0f0$\xc1\xf5^\r\x86\xfc\x15\xd8\xb1\x13\xfb\x08D\x1a\x0e\xb1|\r8\t\xb8\x07\x0b\x98\x07\x00\xb8b|\xe0\xbf`\x08xi~\xbf_\xaf\x11\xa0\xef"\xdb\x96\xe9~\x13\x00\x0f\xf1L\xbc{'
            
