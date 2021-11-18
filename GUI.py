#!/usr/bin/python3
# Kai Zumpfe
# Leipzig University
# Medical Faculty
# Insitute for Medical Physics and Biophysics

import numpy as np
from os import system, listdir, mkdir
from os.path import join, exists
from sys import platform
from subprocess import Popen
from calcs import *
from Full_Analysis import KaiMarkov
import tkinter as tk
import sys
sys.path.append('/Users/albertsmith/Documents/GitHub')
import pyDIFRATE as DR
from pyDIFRATE.data.load_nmr import load_NMR
from MolSystem import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib

matplotlib.rcParams.update({"lines.linewidth": 1,
                            "axes.labelsize": 8,
                            "xtick.labelsize": 8,
                            "ytick.labelsize": 8,
                            "axes.titlesize": 10,
                            'font.size': 8})
TEXTBOX = {'facecolor': 'lightblue',
           'alpha': 0.5}


class SubFrame(tk.Frame):
    '''This class is used to be the frame for everything we actually want to show, representing a full page, but additional
    for smaller frames on these pages. Creation and deleting should basically work recursivele over
    self.create and self.forget function'''
    def __init__(self,parent):
        tk.Frame.__init__(self,master=parent)
        self.create()
    def create(self):
        '''when youn inherit from this frame, here you create your objects like labels, buttons, canvas" or whatever
        the hell you want'''
        pass

    def update_me(self):
        '''this function does not have a use right now, it was planned for updating plots or smth, but I am not sure
        if we need it'''
        pass

    def get_from_master(self,attribute):
        if hasattr(self,"master"):
            return self.master.get_from_master(attribute)
        elif hasattr(self,attribute):
            return getattr(self, attribute)
        else:
            print("Attr. not available")

    def set_at_master(self,attribute,value):
        if hasattr(self,"master"):
            self.master.set_at_master(attribute, value)

    def destroy(self):
        """if you change your mainpage, the old page should be forgotten or deleted, including all buttons, labels etc.
        that were created in this frame"""
        #todo be careful to not have a memory leak
        for child in self.children.copy():
            self.children[child].destroy()
        tk.Frame.destroy(self)


class Simulations(SubFrame):
    """a little panel that should display all available simulations in your "xtc" folder"""
    name="Simulations"
    def create(self):
        if not exists("xtcs"):
            mkdir("xtcs")
        sims = [f for f in listdir("xtcs/") if f.endswith(".xtc")]
        for i,f in enumerate(sims):
            tk.Label(master=self, text=f).grid(row=i)


class PDBs(SubFrame):
    """a little panel that should display all available pdbs in your "xtc" folder"""
    name="PDBS"
    def create(self):
        def open_chimera(f):
            if "linux" in platform:
                chim_path = "chimerax"
            else:
                chim_path="/Applications/ChimeraX-1.0.app/Contents/MacOS/ChimeraX"
            Popen([chim_path,"{}".format(join("pdbs",f))])
        if not exists("pdbs"):
            mkdir("pdbs")
        pdbs = [f for f in listdir("pdbs")]
        for i,f in enumerate(pdbs):
            tk.Button(master=self, text=f, command=lambda f=f:open_chimera(f)).pack()

class Plot_MD_Analysis(SubFrame):
    name="MDAnalysisFrame"
    def __init__(self,parent,*args):
        self.add_pages(*args)
        SubFrame.__init__(self,parent)

    def add_pages(self,*args):
        if not hasattr(self,"pages"):
            self.pages = []
        for arg in args:
            print("added page")
            self.pages.append(arg)

    def create(self):
        def plot_page(p):
            if hasattr(self,"canvas"):
                self.canvas.get_tk_widget().destroy()
                self.toolbar.destroy()

            self.canvas = FigureCanvasTkAgg(p, master=self)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, expand=1)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.canvas.get_tk_widget().config(width=1000,height=600)
            #hbar = tk.Scrollbar(self,self.canvas,orient=tk.HORIZONTAL)

        for i,p in enumerate(self.pages):
            tk.Button(self,text="Page "+str(i), command=lambda p= p: plot_page(p)).pack(side=tk.TOP)#grid(column=i,row=0)

class MolDynFrame(SubFrame):
    """Here I want to directly access my MD simulation data with processing settings and so on, maby we can direclty
    output the resulting detector analysis then here"""
    name="Molecular Dynamics"
    def create(self):
        def get_residue_analysis():
            res = self.MethylBox.selection_get()
            print(res)
            self.Plot_Frame = Plot_MD_Analysis(self, *self.M.plot_det_gui(res))
            self.Plot_Frame.grid(column=1,row=1,rowspan=5)
            #todo create a canvas


        def analysis():
            #remove old listbox with methyl_groups(if available)
            if hasattr(self,"MethylBox"):
                self.MethylBox.destroy()
                self.AButton.destroy()  #todo find out if hiding/unhiding might be better -K

            #todo this is sill garbage, just a functional test right now -K
            sim = self.MD_Listbox.selection_get()
            OutLabel.config(text= "residues of "+sim+":")
            self.M = KaiMarkov(sel_sim=join("xtcs",sim))
            self.set_at_master("MD_Sim_Analysis",self.M)
            self.MethylBox = tk.Listbox(self)
            for i,key in enumerate(self.M.full_dict.keys()):
                self.MethylBox.insert(i,key)
            self.MethylBox.grid(column=0,row=4)
            self.AButton = tk.Button(self,text="Analyse Residue", command= lambda: get_residue_analysis()).grid(column=0,row=5)
            self.M.calc()


        tk.Label(self,text="available simulations").grid(column=0,row=0)
        self.MD_Listbox = tk.Listbox(self)
        for i,f in enumerate([f for f in listdir("xtcs") if f.endswith(".xtc")]):
            self.MD_Listbox.insert(i, f)
        self.MD_Listbox.grid(column=0,row=1)
        tk.Button(self,text="Get methyl cont. residues", command=lambda:analysis()).grid(column=0,row=2)
        OutLabel = tk.Label(self)
        OutLabel.grid(column=0,row=3)

class DetectorFrame(SubFrame):
    '''this panel should display the detector senstitivities and detector responses in the NMR panel. right now it is just
    working for HET-s, because the creation of detectros and the selection is hard coded. It should get access to the upper frame
    and get values from boxes to know about number of detectors, the residiues etc'''
    name="Detectors"
    def create(self):
        het = load_NMR(self.get_from_master("nmr_file"))
        het.del_data_pt(range(12, 17))
        het.label = [lab.replace("Val,", ",") for lab in het.label]
        het.label = [lab.replace("Leu,", ",") for lab in het.label]
        het.detect.r_auto(5, inclS2=True, Normalization='MP')
        fit = het.fit()
        fig = plt.figure()
        #todo rearrange the plot functionality, gives some problems with the GUI on linux -K
        n_dets = self.master.n_detectors.get()

        fit.plot_rho(fig=fig, style="bar", rho_index=range(int(n_dets)), errorbars=True)
        fig.set_size_inches(8,6)

        canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.get_tk_widget().config(width=800,height=500)


class NMRFrame(SubFrame):
    '''In this frame you should be able to view the detector analysis for your NMR data, right now just properly working
    for HET-s'''
    name="NMR"
    def create(self):
        def load_data(folder,f):
            self.set_at_master("nmr_file",join("nmr",folder,f))

        def get_data(folder):
            for i,f in enumerate(listdir(join("nmr",folder))):
                tk.Button(self, text=f, command=lambda f= f, folder=folder: load_data(folder, f)).grid(column=1, row=i)

        #TODO remove label when everything is wokring
        tk.Label(self, text="Warning, just working with CDH2 data of HET's right now, because hard coded").grid(column=1,row=1)

        # Creating a Button for every system available in NMR folder, now HET-s, lipid, ubiquitin?
        for i,folder in enumerate(listdir("nmr")):
            tk.Button(self,text=folder,command=lambda f=folder: get_data(f)).grid(column=0, row=i)

        self.n_detectors = tk.StringVar(self)
        tk.Label(self,text="Number of detectors:").grid(column=2,row=2)
        tk.Entry(self, textvariable=self.n_detectors).grid(column=3, row=2)



        # Creating a Subframe for plotting detector analysis in this frame
        def plot_detector():
            '''creating a Frame to Plot Detectors according to the selected NMR file in master.nmr_file'''
            DetectorFrame(self).grid(column=1,row=4,sticky=tk.W)
        tk.Button(self, text="Detector Analysis", command = lambda: plot_detector()).grid(column=0, row=10)


class PlotFrame(SubFrame):
    '''Example frame how to implement a matplotlib canvas into tkinter'''
    name = "Plot"
    def create(self):
        def plot_whatever():
            ax.cla()
            ax.plot([0, 1, 2, 3, 0, 1, 2, 0, 1, 0])
            canvas.draw()
        def plot_B():
            ax.cla()
            ax.plot(t, 2 * np.sin(2 * np.pi * t))
            canvas.draw()
        fig = plt.Figure(figsize=(7, 5), dpi=100)
        t = np.arange(0, 3, .01)
        ax = fig.add_subplot(211)
        bx = fig.add_subplot(212)
        ax.plot(t, 2 * np.sin(2 * np.pi * t))
        bx.plot([3,1,3,2,3,4,1,2,3,1,4])
        canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #def on_key_press(event):
        #    print("you pressed {}".format(event.key))
        #    key_press_handler(event, canvas, toolbar)
        #canvas.mpl_connect("key_press_event", on_key_press)

        tk.Button(self, text="what", command=lambda: plot_whatever()).pack()
        tk.Button(self, text="ever", command=lambda: plot_B()).pack()

class FirstFrame(SubFrame):
    name="Mainpage"
    def create(self):
        tk.Label(self,text="Welcome to pyDIFRATE, your friendly helper for analyzing and interpreting NMR relaxation data"
                                 "in coorperation with MD Simulation data").pack()
        Simulations(self).pack(side=tk.LEFT)
        PDBs(self).pack(side=tk.RIGHT)

class ParentFrame(tk.Tk):
    """This is the front page of our program, everything important for the gui will be initialized here, also all
    important data/values should be stored in the class attributes"""
    #Todo find a better name for it
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry("1280x720")
        self.subframe = None
        self.pages = [FirstFrame,NMRFrame,MolDynFrame]
        for i, frameobject in enumerate(self.pages):
            '''creating a button for every Page the program should have. Pages are represented by a class inheriting from
            SubFrame'''
            #TODO I am thinking about putting the Frame definition in a separate file/folder and just iterate over
            #TODO the things inside to load every possible page here
            tk.Button(self, text=frameobject.name,
                      command=lambda x = frameobject: self.load_subframe(x)).grid(row=1, column=i, sticky=tk.N)
            #todo somehow fix the columns it is annoying that the jump pending on the selected page -K
        self.load_subframe(FirstFrame)  # initialising the mainpage
        #self.pack()  #every object must be packed (.pack()) or set in .grid()
        # every method has her advantages, but you cannot mix both in the same frame
        #self.rowconfigure(0,weight=1)
        #self.columnconfigure(0,weight=1)

        self.mainloop()

    def load_subframe(self, frame_object):
        """if a frame is already loaded (which usually should be the case), let us destroy the frame and its contents"""
        if self.subframe is not None:
            self.subframe.destroy()
        self.subframe = frame_object(self)
        #todo find out if i might be able to define a size in pixels or smth like this -K
        self.subframe.grid(row=2, column=0, columnspan=len(self.pages))

    def get_from_master(self, attribute):
        '''since the frames are sometimes very nested, this function will get an attribut from the actual uppest parent
        (this one here) and return it down to the (grand(grand))child, for example to store the visualized simulation
        or system and different settings that might be useful'''
        if hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            print("Attr. not available")

    def set_at_master(self,attribute,value):
        '''basically the inverse of the get_from_master function'''
        setattr(self, attribute, value)

if __name__ == '__main__':
    ParentFrame()