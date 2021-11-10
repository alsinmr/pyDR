#!/usr/bin/python3
# Kai Zumpfe
# Leipzig University
# Medical Faculty
# Insitute for Medical Physics and Biophysics

import numpy as np

from os import system, listdir, mkdir
from os.path import join,  exists
from subprocess import Popen
import tkinter as tk
import pyDIFRATE as DR
from pyDIFRATE.data.load_nmr import load_NMR
from MolSystem import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib

matplotlib.rcParams.update({"lines.linewidth": 1,
                            "axes.labelsize": 8,
                            "xtick.labelsize": 8,
                            "ytick.labelsize": 8,
                            "axes.titlesize": 10,
                            'font.size': 8})
TEXTBOX = {'facecolor': 'lightblue',
           'alpha': 0.5}


class SubFrame():
    '''This class is used to be the frame for everything we actually want to show, representing a full page, but additional
    for smaller frames on these pages. Creation and deleting should basically work recursivele over
    self.create and self.forget function'''
    def __init__(self,parent, root):
        self.parent = parent
        self.frame = tk.Frame(root)
        self.objects = []
        self.create()
    def create(self):
        '''when youn inherit from this frame, here you create your objects like labels, buttons, canvas" or whatever
        the hell you want'''
        pass

    def update_me(self):
        '''this function does not have a use right now, it was planned for updating plots or smth, but I am not sure
        if we need it'''
        pass

    def forget(self):
        '''if you change your mainpage, the old page should be forgotten or deleted, including all buttons, labels etc.
        that were created in this frame'''
        #todo be careful to not have a memory leak
        for child in self.frame.children.copy():
            self.frame.children[child].destroy()
        self.frame.forget()

    def pack(self,**kwargs):
        '''just for forwarding the kwargs to the original pack function'''
        self.frame.pack(kwargs)

    def grid(self,**kwargs):
        '''just forwarding the kwargs to the original grid funciton'''
        self.frame.grid(kwargs)

class Simulations(SubFrame):
    '''a little panel that should display all available simulations in your "xtc" folder'''
    name="Simulations"
    def create(self):
        if not exists("xtcs"):
            mkdir("xtcs")
        sims = [f for f in listdir("xtcs/") if f.endswith(".xtc")]
        for i,f in enumerate(sims):
            tk.Label(master=self.frame,text=f).grid(row=i)

class PDBs(SubFrame):
    '''a little panel that should display all available pdbs in your "xtc" folder'''
    name="PDBS"
    def create(self):
        def open_chimera(f):
            Popen(["chimerax","{}".format(join("pdbs",f))])
        if not exists("pdbs"):
            mkdir("pdbs")
        pdbs = [f for f in listdir("pdbs")]
        for i,f in enumerate(pdbs):
            tk.Button(master=self.frame, text=f, command=lambda f=f:open_chimera(f)).pack()


class DetectorFrame(SubFrame):
    '''this panel should display the detector senstitivities and detector responses in the NMR panel. right now it is just
    working for HET-s, because the creation of detectros and the selection is hard coded. It should get access to the upper frame
    and get values from boxes to know about number of detectors, the residiues etc'''
    name="Detectors"
    def create(self):
        #def plot_het():
        h = load_NMR(self.parent.nmr_file)
        #h.load(filename=self.parent.nmr_file)#get_nmr_experiment(self.parent.nmr_file.split(".")[-2].split("/")[-1])
        h.del_data_pt(range(12, 17))

        h.label = [lab.replace("Val,", ",") for lab in h.label]
        h.label = [lab.replace("Leu,", ",") for lab in h.label]
        resids = [81, 84, 90, 92, 94, 97, 100, 101, 117, 128, 129, 130]
        #h.molecule.load_structure_from_file_path(n=0)
        #h.molecule.select_atoms(Nuc="ivlat", resids=resids, segids="B")
        #h.molecule.sel1 = h.molecule.sel1[::3]
        #h.molecule.sel2 = h.molecule.sel2[::3]

        h.detect.r_auto(5, inclS2=True, Normalization='MP')

        e = h.fit()
        fig = plt.figure()
        e.plot_rho(fig=fig,style="bar", rho_index=range(4), errorbars=True)
        fig.set_size_inches(5,4)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.frame)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, self.frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


class NMRFrame(SubFrame):
    '''In this frame you should be able to view the detector analysis for your NMR data, right now just properly working
    for HET-s'''
    name="NMR"
    def create(self):
        def load_data(folder,f):
            self.parent.nmr_file = join("nmr",folder,f)
            print(self.parent.nmr_file)

        def get_data(folder):
            for i,f in enumerate(listdir(join("nmr",folder))):
                tk.Button(self.frame,text=f,command=lambda f= f,folder=folder: load_data(folder,f)).grid(column=1,row=i)

        #TODO remove label when everything is wokring
        tk.Label(self.frame, text="Warning, just working with CDH2 data of HET's right now, because hard coded")

        # Creating a Button for every system available in NMR folder, now HET-s, lipid, ubiquitin?
        for i,folder in enumerate(listdir("nmr")):
            tk.Button(self.frame,text=folder,command=lambda f=folder: get_data(f)).grid(column=0,row=i)


        # Creating a Subframe for plotting detector analysis in this frame
        def plot_detector():
            DetectorFrame(self.parent,self.frame).grid(column=2,row=10)
        tk.Button(self.frame,text="Detector Analysis", command = lambda: plot_detector()).grid(column=0,row=10)





class PlotFrame(SubFrame):
    '''Example frame how to implement a matplotlib canvas into tkinter'''
    name = "Plot"
    def create(self):
        def plot_whatever():
            ax.cla()
            ax.plot([0,1,2,3,0,1,2,0,1,0])
            canvas.draw()
        def plot_B():
            ax.cla()
            ax.plot(t, 2 * np.sin(2 * np.pi * t))
            canvas.draw()
        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, .01)
        ax = fig.add_subplot(211)
        bx = fig.add_subplot(212)
        ax.plot(t, 2 * np.sin(2 * np.pi * t))
        bx.plot([3,1,3,2,3,4,1,2,3,1,4])
        canvas = FigureCanvasTkAgg(fig, master=self.frame)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, self.frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        def on_key_press(event):
            print("you pressed {}".format(event.key))
            key_press_handler(event, canvas, toolbar)

        canvas.mpl_connect("key_press_event", on_key_press)

        def _quit():
            root.quit()  # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        tk.Button(master=self.frame,text="what",command=lambda: plot_whatever()).pack()
        tk.Button(master=self.frame,text="ever",command=lambda: plot_B()).pack()
        button = tk.Button(master=self.frame, text="Quit", command=_quit)
        button.pack(side=tk.BOTTOM, fill=tk.BOTH)

class FirstFrame(SubFrame):
    name="Mainpage"
    def create(self):
        tk.Label(self.frame,text="Welcome to pyDIFRATE, your friendly helper for analyzing and interpreting NMR relaxation data"
                                 "in coorperation with MD Simulation data").pack()
        Simulations(self.parent, self.frame).pack(side=tk.LEFT)
        PDBs(self.parent, self.frame).pack(side=tk.RIGHT)

class ParentFrame():
    def __init__(self, root):

        self.frame = tk.Frame(root)
        self.subframe = None
        self.pages = [FirstFrame,NMRFrame,PlotFrame,PDBs,Simulations]
        for i, frameobject in enumerate(self.pages):
            '''creating a button for every Page the program should have. Pages are represented by a class inheriting from
            SubFrame'''
            tk.Button(self.frame,text=frameobject.name,
                                command=lambda x = frameobject: self.load_frame(x)).grid(row=1,column=i,sticky=tk.W)
        self.load_frame(FirstFrame) # initialising the mainpage
        self.frame.pack()  #every object must be packed (.pack()) or set in .grid()
        # every method has her advantages, but you cannot mix both in the same frame

    def load_frame(self, frame_object):
        #if a frame is already loaded (which usually should be the case), let us 'forget' the frame and its contents
        if self.subframe is not None:
            self.subframe.forget()

        self.subframe = frame_object(self,self.frame)
        self.subframe.grid(row=2,column=0,columnspan=len(self.pages))




if __name__ == '__main__':
    # basic initial settings
    root = tk.Tk()
    root.title("pyDIFRATE by Andy Smith and Kai Zumpfe")
    root.geometry("1280x720")   # size of the gui in pixels, you can fix the size somehow if you want
    mainframe = ParentFrame(root)  # creating mainframe
    root.mainloop()  # < the actual start of the GUI

