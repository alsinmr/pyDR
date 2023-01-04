#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:13:30 2020

@author: albertsmith
"""

import os

#%% Chimera script writing
def chimera_path(**kwargs):
    "Returns the location of the ChimeraX program"
    
    assert is_chimera_setup(),\
        "ChimeraX path does not exist. Run chimeraX.set_chimera_path(path) first, with "+\
        "path set to the ChimeraX executable file location."
    
    with open(os.path.join(get_path(),'ChimeraX_program_path.txt'),'r') as f:
        path=f.readline()
    
    return path

def is_chimera_setup():
    "Determines whether chimeraX executable path has been provided"
    return os.path.exists(os.path.join(get_path(),'ChimeraX_program_path.txt'))

def clean_up():
    """Deletes chimera scripts and tensor files that may have been created but 
    not deleted
    
    (Under ideal circumstances, this shouldn't happen, but may occur due to errors)
    """
    
    names=[fn for fn in os.listdir(get_path()) \
           if fn.startswith('chimera_script') and fn.endswith('.py') and len(fn)==19]
    
    for n in names:
        os.remove(os.path.join(get_path(),n))
    
    print(f'{len(names)} files removed')

def set_chimera_path(path):
    """
    Stores the location of ChimeraX in a file, entitled ChimeraX_program_path.txt
    
    This function needs to be run before execution of Chimera functions (only
    once)
    """
    assert os.path.exists(path),"No file found at '{0}'".format(path)
    
    with open(os.path.join(get_path(),'ChimeraX_program_path.txt'),'w') as f:
        f.write(path)
        

def run_command(**kwargs):
    "Code to import runCommand from chimeraX"
    return 'from chimerax.core.commands import run as rc\n'

def get_path(filename=None):
    """
    Determines the location of THIS script, and returns a path to the 
    chimera_script given by filename.
    
    full_path=get_path(filename)
    """
    dir_path=os.path.dirname(os.path.realpath(__file__))
    return dir_path if filename is None else os.path.join(dir_path,filename)

def WrCC(f,command,nt=0):
    """Function to print chimera commands correctly, using the runCommand function
    within ChimeraX. nt specifies the number of tabs in python to use.
    
    f:          File handle
    command:    command to print
    nt:         number of tabs
    
    WrCC(f,command,nt)
    """
    for _ in range(nt):
        f.write('\t')
    f.write('rc(session,"{0}")\n'.format(command))
    

def py_line(f,text,nt=0):
    """
    Prints a line to a file for reading as python code. Inserts the newline and
    also leading tabs (if nt specified)
    
    python_line(f,text,nt=0)
    """
    
    for _ in range(nt):
        f.write('\t')
    f.write(text)
    f.write('\n')
