#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of pyDIFRATE

pyDIFRATE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyDIFRATE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyDIFRATE.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de



Created on Wed Aug 21 13:21:49 2019

@author: albertsmith
"""


from . import frames
import sys



def new_fun(Type,molecule,**kwargs):
    """
    Creates a function to calculate a particular vector(s) from the MD trajectory.
    Mainly responsible for searching the vec_funs files for available functions and
    returning the appropriate function if found (Type determines which function to use)
    
    Required arguments are Type (string specifying the function to be used) and
    a molecule object (contains the MDAnalysis object)
    
    fun=new_fun(Type,molecule,**kwargs)
    
    """

    fun0=None
    if is_valid(frames,Type):
        fun0=frames.__dict__[Type]
    
    if fun0 is None:
        raise Exception('Frame "{0}" was not recognized'.format(Type))
    
    if len(kwargs)==0:
        print_frame_info(Type)
        return
    
    try:       
        fun=fun0(molecule,**kwargs)
    except:
        print_frame_info(Type)
        assert 0,'Frame definition failed (frame function could not be created),\n'+\
            'Error:{0}, {1}'.format(*sys.exc_info()[:2]) 
    
    frame_index=None
    info={}
    if hasattr(fun,'__len__') and len(fun)==2:fun,frame_index=fun
    if hasattr(fun,'__len__') and len(fun)==3:fun,frame_index,info=fun
    
    try:
        fun()
    except:
        assert 0,'Frame function failed to run, ,\n'+\
            'Error:{0}, {1}'.format(*sys.exc_info()[:2])     
    
    return fun,frame_index,info

def return_frame_info(Type=None):
    """
    Provides information as to what frames are available, and what arguments they
    take.
    
    frames=return_frame_info()  Returns list of the frames
    
    args=return_frame_info(Type)    Returns argument list and help info for Type
    """
    
    if Type is None:
        fun_names=list()
        for n in dir(frames):
            if is_valid(frames,n):
                fun_names.append(n)
        return fun_names
    else:
        if is_valid(frames,Type):
            code=frames.__dict__[Type].__code__
            return code.co_varnames[1:code.co_argcount]

        print('Frame "{0}" is not implemented'.format(Type))
        return
            
def print_frame_info(Type=None):
    """
    Prints out some information about the possible frames
    """
    
    if Type is None:
        fun_names=return_frame_info()
        print('Implemented frames are:')
        for f in fun_names:
            args=return_frame_info(f)
            print('"{0}" with arguments {1}'.format(f,args))
    else:
        args=return_frame_info(Type)
        if args is not None:
            print('"{0}" has arguments {1}'.format(Type,args))
    
def is_valid(module,Type):
    """
    Checks if a frame is included in a given module
    """
    return Type in dir(module) and hasattr(module.__dict__[Type],'__code__') and\
         module.__dict__[Type].__code__.co_varnames[0]=='molecule'