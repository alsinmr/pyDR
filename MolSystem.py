import os,sys

from os import listdir
from os.path import join, dirname
#this with sys path is little annoying but the only way to make it usable from outer folder right now
sys.path.append(dirname(__file__))
sys.path.append(join(dirname(__file__),"difrate"))


#todo the imports are not working properly right now

import MDAnalysis as MDA
from pyDIFRATE.Struct.structure import *
#import pyDIFRATE.
#from pyDIFRATE.data_class import *
import shutil
import matplotlib.pyplot as plt
import numpy as np
import hashlib
#from pyDIFRATE.Ctsens import Ct
#from pyDIFRATE import submodules,Ct_fast
from scipy.stats import pearsonr
from time import time


sheet = ["228Ala","231Ile","237Ala","239Val","241Leu","264Val","267Val","268Val","276Leu","277Ile"]
loop = ["244Val","247Ala","248Ala","250Leu"]
inner = ["228Ala","231Ile","237Ala","239Val","241Leu","264Val","267Val","277Ile"]
outer = ["244Val","247Ala","248Ala","250Leu","268Val","276Leu"]


def hash(filename):
  h = hashlib.sha256()
  b  = bytearray(128*1024)
  mv = memoryview(b)
  with open(filename, 'rb', buffering=0) as f:
    for n in iter(lambda : f.readinto(mv), 0):
        h.update(mv[:n])
  return h.hexdigest()

subs_fold = join(dirname(__file__),"Substance_data/")


class MolSystem():

    '''This ist the main Object, that should contain every experimental and/or structural data object'''
    def __init__(self,name):
        '''To Initialize just type in the Name of your experiment. If data are already available in the 'Substance_data/' folder,
        they will be immediately loaded. Besides a new experiment folder will be created '''
        self.name = name
        self.folder = None
        self.molecule = Molecule(self)
        #dict for DataCont() Objects to process NMR data
        self.nmr_raw_data = {}
        self.nmr_selected = None
        self.selected_raw_nmr = None
        self.nmr_processed_data = {}
        self.md_data = {}
        self.md_sel = None
        self.fret_data = {}
        if len(name):
            if name in os.listdir(subs_fold):
                self.load_from_folder()
            else:
                self.create_new_experiment(name)

    def add_structure_file(self,filename):
        '''loading the structure by filename (.pdb file")
        since the function will copy the pdb file to you experimental data folder, the name of the experiment should already been given'''
        assert len(self.name),"Please give your Experiment a name"
        #check these function, be careful by passing filename=filename by kwargs (not working for some reason)
        if self.molecule is None:
            self.molecule = Molecule()
        self.molecule.load_struct(filename)
        print("Structure data loaded from",filename)
        shutil.copy(filename,join(subs_fold,self.name,filename))

    def add_nmr_experiment(self,filename):
        '''Copying a file to your experimental data folder'''
        assert len(self.name), "Please give your Experiment a name"
        print("NMR file added to you experimental data")
        shutil.copy(filename, join(subs_fold,self.name,"NMR",filename))

    def select_simulation(self, number = None):
        simulations = [f for f in listdir(join(self.folder, "MDSimulation")) if f.endswith(".xtc")]
        if number is None:
          print("Available Simulations:")
          for i, sim in enumerate(simulations):
            print(i, sim)
          i = int(input("Choosen: "))
        else:
          i = number
        try:
            self.xtc = join(self.folder, "MDSimulation", simulations[i])
            self.xtc_fn = simulations[i]
        except:
            print("not a valid simulation")
            self.select_simulation()

        for pdb in listdir(join(self.folder, "pdbs")):
            f = join(self.folder, "pdbs", pdb)
            print(f)
            try:
                uni = MDA.Universe(f, self.xtc)
                self.pdb = f
                break
            except:
                uni = None

        assert uni, "Couldnt find a fitting pdb for this simulation, please copy one in the pdb folder"
        print("Simulation", self.xtc_fn, "and", self.pdb, "selected")
        return len(uni.segments)

    def compare(self,**kwargs):
        '''This function should compare different data from MDSimulation/NMR/FRET
        Since there is a very high chance, that, depending on your datasets, this function is not gonna work
        automatically, I recommend to create a new class inheriting from this one and overload the
        compare function with your specifications example below (HETs_CH3, HETs_NH)'''
        sum_experiments = len(self.nmr_raw_data.keys()) + len(self.md_data.keys())
        assert sum_experiments > 1, "You should provide more than 1 dataset to compare anything"
        exp = self.get_nmr_experiment()
        if "Nuc" in kwargs:
            Nuc = kwargs["Nuc"]
        else:
            Nuc = input("specify nucleus: (ivla,N)")
        ##print out more information what is happening here
        if "motion" in kwargs:
            motion = int(kwargs["motion"])
        else:
            motion = int(input("motion (0,1,2,3)"))

        self.molecule.load_structure_from_file_path()
        self.molecule.select_atoms(Nuc=Nuc)
        exp.detect.r_auto(5 if not "n_det" in kwargs else int(kwargs["n_det"]), inclS2=True, Normalization="MP")
        fit = exp.fit()
        ax = fit.plot_rho(rho_index=range(3), errorbars=True)
        targ = fit.sens.rhoz()[:3]
        targ[0, 84:] = 0
        frame_A = "bond"
        frame_B = "chain_rotate"
        n_mds = self.n_simulations
        for i in range(n_mds if "all" in kwargs and kwargs['all'] else 1):
            self.load_MD(i if "all" in kwargs and kwargs['all'] else None)

            self.molecule.tensor_frame(Type=frame_A, Nuc=Nuc,segids= 'B')

            self.molecule.new_frame(Type=frame_B, Nuc=Nuc,segids = 'B')
            self.md_data[self.xtc_fn] = difrate.eval_fr.frames2data(self.molecule, n=-1 if not 'n_frames' in kwargs else kwargs['n_frames'])
            self.molecule.select_atoms(Nuc=Nuc, resids=fit.label)
            md = self.md_data[self.xtc_fn][motion]
            md.detect.r_auto(target=targ, n=8)
            md = md.fit()
            if Nuc == 'ivla':
                for x, aa in enumerate(exp.label):
                    if "Ala" in aa:
                        md.average_data_pts([x, x + 1, x + 2])
                    elif "Ile" in aa or "Val" in aa or "Leu" in aa:
                        md.average_data_pts([x, x + 1, x + 2])
                        md.average_data_pts([x + 1, x + 2, x + 3])
                        md.choose_better_met(fit, [x, x + 1])
                        # TODO   update selection if Ile because only the terminal group is deuterated

        x0 = range(md.R.shape[0])
        for sim in self.md_data.keys():
            for k, a in enumerate(ax):
                a.plot(x0, self.md_data[sim].R[:, k], linestyle="dashed", linewidth=2)
        plt.show()

    def load_nmr_experiment_from_file(self,filename:str):
        '''Adding a nmr experiment file to your Substance project
        Checks for the file extension and deciding what loading function will be used
        can use the usual one for txt files
        and the matlab-function for mat files'''
        name = filename.rsplit(".")[0].rsplit("/")[-1]
        self.nmr_raw_data[name] = DataCont(self)

        file_ext = filename.rsplit(".")
        if file_ext[1] == "txt":
           self.nmr_raw_data[name].load(filename=filename)
        elif file_ext[1] == "mat":
            self.nmr_raw_data[name].load_from_matlab(filename=filename)
        print("File",filename,"loaded as NMR data for",self.name)
        ### yeah the file loading should in my opionion also be done in the data object itself
        ### doesnt need to be here. just create the object here and add it to the dict

    def load_MD(self,**kwargs):
        '''Loading an MDSimulation from the MD folder in your Substance data. If the data is processed with tf=None
        and n_sparse = -1, the data is stored as float16 (to reduce disk-space, if you want to change go to
        submodules.__init__.Defauults)'''
        #need to reload the molecule object
        self.molecule = Molecule(self)
        frame_A = kwargs["frame_A"] if "frame_A" in kwargs else "bond"
        frame_B = kwargs["frame_B"] if "frame_B" in kwargs else "chain_rotate"
        Nuc = kwargs["Nuc"] if "Nuc" in kwargs else "ivla"
        resids = kwargs.get("resids")
        #segids = kwargs.get("segids")
        tf = kwargs.get("tf")
        n_sparse = kwargs["n_sparse"] if "n_sparse" in kwargs else -1
        reload = True if "force_reload" in kwargs and kwargs['force_reload'] else False
        if self.select_simulation(kwargs.get("sim")) == 3: #TODO find a better place for this
            segids="B"
        else:
            segids= "C"

        self.molecule.load_struct(self.pdb, self.xtc)
        self.molecule.select_atoms(Nuc=Nuc, resids=resids, segids=segids)
        self.molecule.tensor_frame(Type= frame_A, Nuc = Nuc, resids = resids, segids = segids)
        #self.molecule.tensor_frame(sel1=1,sel2=2)
        if frame_B == "peptide_plane":
            self.molecule.new_frame(Type=frame_B, resids=resids, segids=segids)
        else:
            self.molecule.new_frame(Type= frame_B,  Nuc = Nuc, resids = resids, segids = segids)#,frame_index=np.arange(44).repeat(3))

        dump = True if not reload else False
        calc = True

        md_file_hash = hash(self.xtc)
        npy = join(self.folder,"MDSimulation",self.xtc.rsplit(".")[0])
        npy += Nuc + "_"
        if frame_A == "bond":
            npy += "b_"
        if frame_B == "chain_rotate":
            npy+= "cr"
        elif frame_B== "bond_rotate":
            npy+= "br"
        elif frame_B == "librations":
            npy += "libr"
        elif frame_B == "peptide_plane":
            npy+="pp"
        elif frame_B == "methylCC":
            npy+= "meCC"

        if os.path.exists(npy+".npy") and not reload:
            dat = np.load(npy+".npy",allow_pickle=True).item()
            if dat["md_file_hash"] == md_file_hash:
                #TODO also check if frame_A and frame_B and so on and so on are fitting, besides it could become a problem
                #TODO maybe insert shortcuts into the filename for that purpose, but then you end up with multiple files

                dump=False
                if not tf and n_sparse<0:
                  calc=False
                  print("Simulation already processed and gonna be loaded from temp file")
            else:
                print("Simulation is new or has been changed, has to be processed")
            data = []
            #print(dat.keys())
            for key in dat.keys():
                if not 'data' in key:
                    continue  #exclude the keys that not contain the data
                d = DataCont(self)
                for key2 in dat[key].keys():
                    setattr(d,key2,dat[key][key2])
                d.sens = Ct(d,t=dat['data0']['tpoints'])
                d.new_detect()
                data.append(d)

            self.md_data[self.xtc_fn] = data


        if calc:
            if tf or n_sparse>0:
                dump = False
            self.md_data[self.xtc_fn] = difrate.eval_fr.frames2data(self.molecule,
                                                                    n=n_sparse, tf=tf)  # to load_md
        if dump:
            #TODO this looks not really nice, make it nicer     -K

            dat = {}
            dat.update(Nuc=Nuc)
            dat.update(frame_A=frame_A)
            dat.update(frame_B = frame_B)
            dat.update(resids = resids)
            dat.update(segids = segids)
            dat.update(pdb_file = self.molecule.structure_file_path)
            dat.update(md_file_hash=md_file_hash)
            ##list of dicts bzw numpy array of dicts
            data0 = {}
            data1 = {}
            data2 = {}
            data3 = {}
            data0.update(R=self.md_data[self.xtc_fn][0].R)
            data0.update(R_std=self.md_data[self.xtc_fn][0].R_std)
            data0.update(tpoints = self.md_data[self.xtc_fn][0].sens.t())
            #also the standard deviation of the sens object?    K
            #maybe s2 of vars?
            data1.update(R=self.md_data[self.xtc_fn][1].R)
            data1.update(R_std=self.md_data[self.xtc_fn][1].R_std)

            data2.update(R=self.md_data[self.xtc_fn][2].R)
            data2.update(R_std=self.md_data[self.xtc_fn][2].R_std)
            data2.update(R_l = self.md_data[self.xtc_fn][2].R_l)
            data2.update(R_u = self.md_data[self.xtc_fn][2].R_u)
            data2.update(Rc = self.md_data[self.xtc_fn][2].Rc)
            data2.update(Rin = self.md_data[self.xtc_fn][2].Rin)
            data2.update(Rin_std = self.md_data[self.self.xtc_fn][2].Rin_std)
            data2.update(chi2 = self.md_data[self.xtc_fn][2].chi2)

            data3.update(R=self.md_data[self.xtc_fn][3].R)
            data3.update(R_std=self.md_data[self.xtc_fn][3].R_std)
            for key in data0.keys():
                data0[key] = np.array(data0[key],dtype=submodules.Defaults.df_store)
            for key in data1.keys():
                data1[key] = np.array(data1[key],dtype=submodules.Defaults.df_store)
            for key in data2.keys():
                data2[key] = np.array(data2[key],dtype=submodules.Defaults.df_store)
            for key in data3.keys():
                data3[key] = np.array(data3[key],dtype=submodules.Defaults.df_store)
            dat.update(data0=data0)
            dat.update(data1=data1)
            dat.update(data2=data2)
            dat.update(data3=data3)
            np.save(npy,dat)
        return self.md_data[self.xtc_fn]

    def create_new_experiment(self,name):
        if name in os.listdir(subs_fold):
            print("Warning! Experiment name already existing. Do you want to proceed?")
        self.name = name
        print("New Experiment with name",name,"created in 'Substance_Data/' Folder")
        self.folder = join(subs_fold,name)
        os.mkdir(self.folder)
        os.mkdir(join(self.folder,"NMR"))
        os.mkdir(join(self.folder, "MDSimulation"))
        os.mkdir(join(self.folder, "FRET"))

    def load_from_folder(self):
        '''loads all available experimental and structural data from the folder of the experiment if available'''
        if len(self.name) == 0:
            print("Please type name of your substance")
            print(os.listdir("Substance_data"))
            self.name = input()
        if self.name in os.listdir(subs_fold):
            self.folder = join(subs_fold,self.name)
        else:
            print("Data from name not available, check for typos")
        '''Load NMR Data'''
        print(self.folder)
        self.n_simulations = len([f for f in os.listdir(join(self.folder,"MDSimulation")) if "xtc" in f.rsplit(".")[1]])
        for f in os.listdir(join(self.folder,"NMR")):
            self.load_nmr_experiment_from_file(join(join(self.folder,"NMR"), f))

    def get_nmr_experiment(self,key=None):
        '''Returning the adress to the nmr-object with the name given in the arguments'''
        if key == None:
            available = [key for key in self.nmr_raw_data]
            if len(available)==1:
                print("Since only 1 experiment is available",available,"is automatically selected")
                key = available[0]
            else:
                print("Please select experimental data\nAvailable experiments:")
                print([(x,av) for x,av in enumerate(available)] )
                key = available[int(input("N="))]
        self.selected_raw_nmr = key
        print(self.nmr_raw_data)
        return self.nmr_raw_data[key]

    def select_atoms(self,**kwargs):
        #denke das kann weg
        '''Make atom selection in the molecule object'''
        if self.molecule == None:
            self.load_molecule_from_structure_data()
        print(self.molecule)
        #this should be changeable by number of experiments i guess
        self.molecule.select_atoms(Nuc="N", resids=self.nmr_raw_data["exp_0"].label,**kwargs)

class HETs_CH3(MolSystem):
    def compare_residue(self,resnum=248,**kwargs):
        sum_experiments = len(self.nmr_raw_data.keys()) + len(self.md_data.keys())
        assert sum_experiments > 1, "You should provide more than 1 dataset to compare anything"
        # exp = self.get_nmr_experiment("13CD2H_data")
        exp = self.get_nmr_experiment()
        tf = kwargs["tf"] if "tf" in kwargs else None
        dummys = kwargs["dummys"] if "dummys" in kwargs else 12
        average = True
        Nuc = kwargs['Nuc'] if 'Nuc' in kwargs else "ivla"
        seg = kwargs['seg'] if 'seg' in kwargs else "B"
        mot = kwargs['mot'] if 'mot' in kwargs else 2
        n_det = kwargs['n_det'] if 'n_det' in kwargs else 3
        n_comp = kwargs["n_comp"] if "n_comp" in kwargs else 2
        legend = []

        self.molecule.load_structure_from_file_path(0)  #
        self.molecule.select_atoms(Nuc=Nuc, segids=seg)  #

        exp.del_data_pt(range(13, 18))
        m = len(exp.label)
        del_range = []
        for n,lab in enumerate(exp.label[::-1]):
            if not str(resnum) in lab:
                exp.del_data_pt(m-n-1)
                del_range.append(m-n-1)
        del_range.sort()
        print(del_range)
        exp.detect.r_auto(5, inclS2=True, Normalization="MP")
        fit = exp.fit()
        fig = plt.figure()
        bx = [fig.add_subplot(4,1,i) for i in range(1,5)]

        residsMD = resnum-219 + 72#[int(l[:3]) - 219 + 72 for l in fit.label]
        ax0, ax = fit.plot_rho(rho_index=range(n_det), errorbars=True)
        a,b = fit.get_rho(rho_index=range(3))
        bx[0].plot(a,b)

        targ = fit.sens.rhoz()[:n_det]
        targ[0, 84:] = 0

        compare = [(0,1)
            ,(0,2)
            ,(0,3)
            ,(0,4)
            ,(1,0)
                   ]
        #first is the number of the pdb, second is the number of simulation, might break if smth changes
        kwargs.update(resids=residsMD)
        for j, i in compare:#enumerate(range(n_comp)):  # range(n_comp):
            x0 = []
            kwargs['pdb'] = j
            md = self.load_MD(i,**kwargs)  # if "all" in kwargs and kwargs["all"] else None,tf=n,resids=residsMD,Nuc=Nuc,seg=seg)
            #md[mot].del_data_pt(del_range)
            md[mot].detect.r_target(target=targ, n=dummys)
            md[mot] = md[mot].fit()
            if j == 0:  # enough for the first simulation at least if all simulations have the same length
                a, b = md[mot].get_rho(rho_index=range(3))
                ax0.plot(a, b, linestyle="dashed")
                bx[0].plot(a,b,linestyle="dashed")
            legend.append(self.xtc_fn)
            # TODO this function is very very slow, do something about it! -K
            for x, aa in enumerate(exp.label):
                print(aa)
                if "Ala" in aa:
                    if average:
                        md[mot].average_data_pts([x, x + 1, x + 2])
                        x0.append(x)
                    else:
                        for _ in range(3):
                            x0.append(x)
                elif "Ile" in aa or "Val" in aa or "Leu" in aa:
                    if average:
                        # print(x,len(self.md_data[self.md_sel][mot].R))#
                        x0.append(x)
                        # x0.append(x)
                        md[mot].average_data_pts([x, x + 1, x + 2])
                        md[mot].average_data_pts([x + 1, x + 2, x + 3])
                        if "Ile" in aa:
                            md[mot].del_data_pt(x)
                        else:
                            x0.append(x)
                    else:
                        for _ in range(6):
                            x0.append(x)
        for k in range(3):
            print(fit.R[:,k])
            bx[k+1].bar(0,height=fit.R[:,k],width=1)
        for l,sim in enumerate(self.md_data.keys()):
            for k, a in enumerate(ax):
                bx[k+1].bar(l+1,height=self.md_data[sim][mot].R[:,k],width=1)

        for k in range(3):
            print(fit.R[:,k])
            real = np.array([fit.R[:,k] for _ in range(-1,len(self.md_data.keys())+1)])
            bx[k+1].plot(real,linestyle="dashed",color="black")

        fig.legend(legend, title="Simulations", bbox_to_anchor=(0, 3.5, 1, 0.0), mode="expand", ncol=4,
                   loc='lower left')
        fig.suptitle("Residue:"+exp.label[0]+" - Nuc: " + Nuc + " - Motion: " + str(mot))
        plt.show()
        exit()
        if "tf" in kwargs and kwargs["tf"]:
            plt.savefig("anim/het" + str(kwargs["tf"]).zfill(5) + ".png")
        else:
            plt.show()

    def compare(self,**kwargs):
        sum_experiments = len(self.nmr_raw_data.keys()) + len(self.md_data.keys())
        assert sum_experiments > 1, "You should provide more than 1 dataset to compare anything"
        #exp = self.get_nmr_experiment("13CD2H_data")
        exp = self.get_nmr_experiment()
        tf = kwargs["tf"] if "tf" in kwargs else None
        dummys = kwargs["dummys"] if "dummys" in kwargs else 12
        average=True
        Nuc = kwargs['Nuc'] if 'Nuc' in kwargs else "ivla"
        seg = kwargs['seg'] if 'seg' in kwargs else "B"
        mot = kwargs['mot'] if 'mot' in kwargs else 2
        n_det = kwargs['n_det'] if 'n_det' in kwargs else 4
        n_comp = kwargs["n_comp"] if "n_comp" in kwargs else 2
        legend = []

        self.molecule.load_structure_from_file_path()#
        self.molecule.select_atoms(Nuc=Nuc, segids=seg)#
        exp.detect.r_auto(5, inclS2=True, Normalization="MP")
        exp.del_data_pt(range(13, 18))
        #exp.draw_rho3D()
        fit = exp.fit()
        residsMD = [int(l[:3]) - 219 + 72 for l in fit.label]
        print(residsMD)
        _, ax0, ax = fit.plot_rho(rho_index=range(n_det), errorbars=True)
        targ = fit.sens.rhoz()[:n_det]
        targ[0, 84:] = 0
        for j, i in enumerate(range(n_comp)):#range(n_comp):
            x0 = []
            md = self.load_MD(None,tf=tf,resids=residsMD,Nuc=Nuc,segids=seg)# if "all" in kwargs and kwargs["all"] else None,tf=n,resids=residsMD,Nuc=Nuc,seg=seg)
            md[mot].detect.r_target(target=targ, n=dummys)
            md[mot] = md[mot].fit()
            if j == 0: #enough for the first simulation at least if all simulations have the same length
              a, b = md[mot].get_rho(rho_index=range(3))
              ax0.plot(a,b, linestyle="dashed")
            legend.append(self.xtc_fn)
            #TODO this function is very very slow, do something about it! -K
            for x, aa in enumerate(exp.label):
                if "Ala" in aa:
                    if average:
                      md[mot].average_data_pts([x, x + 1, x + 2])
                      x0.append(x)
                    else:
                      for _ in range(3):
                        x0.append(x)
                elif "Ile" in aa or "Val" in aa or "Leu" in aa:
                    if average:
                        #print(x,len(self.md_data[self.md_sel][mot].R))#
                        x0.append(x)
                        #x0.append(x)
                        md[mot].average_data_pts([x, x + 1, x + 2])
                        md[mot].average_data_pts([x + 1, x + 2, x + 3])
                        if "Ile" in aa:
                            md[mot].del_data_pt(x)
                        else:
                            x0.append(x)
                        '''
                        #TODO decide if the choosing might be useful in the future    
                        else:
                            md[mot].choose_better_met(fit, [x, x + 1], det_num=0)'''
                    else:
                      for _ in range(6):
                         x0.append(x)
                    # TODO   update selection if Ile because only the terminal group is deuterated
                    # maybe change to scatter plot in that case and jsut plot both methyl groups?   K
        #x0 = range(self.md_data[self.md_sel][mot].R.shape[0])
        for sim in self.md_data.keys():
            for k, a in enumerate(ax):
                #TODO this still looks really weird when not averages, ther must be a better solution
                a.scatter(x0, self.md_data[sim][mot].R[:, k], linestyle="dashed",marker="x", linewidth=1)

        '''
        if average:
            print("Which simulation is closer?")
            for sim in self.md_data.keys():
                print(sim)
                for k in range(len(ax)):
                    print("Detektor:",k)
                    sim_data = self.md_data[sim][mot].R[:, k]
                    fit_data = fit.R[:, k]
                    #print("Pearsonr:",pearsonr(sim_data,fit_data))
                    s = []
                    for num,lab in enumerate(exp.label):
                        d = sim_data[num]/fit_data[num] if sim_data[num] < fit_data[num] else fit_data[num] / sim_data[num]
                        s.append(d if d > 0 else 0)
                    print(np.mean(s))
                print()'''


        plt.legend(legend, title="Simulations", bbox_to_anchor=(0, 3.5, 1, 0.0), mode="expand", ncol=4,
                       loc='lower left')
        plt.suptitle("Nuc: "+Nuc+" - Motion: "+str(mot))

        if "tf" in kwargs and kwargs["tf"]:
          plt.savefig("anim/het"+str(kwargs["tf"]).zfill(5)+".png")
        else:
          plt.show()
    def load_MD(self,n=None,**kwargs):
        '''Loading an MDSimulation from the MD folder in your Substance data. If the data is processed with tf=None
        and n_sparse = -1, the data is stored as float16 (to reduce disk-space, if you want to change go to
        submodules.__init__.Defauults)'''
        #need to reload the molecule object
        self.molecule = Molecule(self)
        #if 'pdb' in kwargs:
        #  self.molecule.load_structure_from_file_path(kwargs['pdb'])
        #else:
        #    self.molecule.load_structure_from_file_path()
        frame_A = kwargs["frame_A"] if "frame_A" in kwargs else "bond"
        frame_B = kwargs["frame_B"] if "frame_B" in kwargs else "chain_rotate"
        Nuc = kwargs["Nuc"] if "Nuc" in kwargs else "ivla"
        resids = kwargs["resids"] if "resids" in kwargs else None
        segids = kwargs["segids"] if "segids" in kwargs else None
        tf = kwargs["tf"] if "tf" in kwargs else None
        n_sparse = kwargs["n_sparse"] if "n_sparse" in kwargs else -1
        reload = True if "force_reload" in kwargs and kwargs['force_reload'] else False
        simulations = [f for f in os.listdir(join(self.folder,"MDSimulation")) if "xtc" in f.rsplit(".")[-1]]
        simulations.sort()
        if n is not None:
            n_sim = n
        elif len(simulations) > 1:
           print("Choose MD-Simulation:")
           for n, f in enumerate(simulations):
              print(n,":",f)
           n_sim = int(input("N="))
        print("Lade",simulations[n_sim])
        self.md_sel = simulations[n_sim]
        md_file_path = join(self.folder,"MDSimulation",simulations[n_sim])
        self.molecule.load_struct(self.molecule.structure_file_path, md_file_path)
        self.molecule.select_atoms(Nuc=Nuc, resids=resids, segids=segids)
        self.molecule.tensor_frame(Type= frame_A, Nuc = Nuc, resids = resids, segids = segids)
        #self.molecule.tensor_frame(sel1=1,sel2=2)
        if frame_B == "peptide_plane":
            self.molecule.new_frame(Type=frame_B, resids=resids, segids=segids)
        else:
            self.molecule.new_frame(Type= frame_B,  Nuc = Nuc, resids = resids, segids = segids)#,frame_index=np.arange(44).repeat(3))

        dump = True if not reload else False
        calc = True

        md_file_hash = hash(md_file_path)
        npy = join(self.folder,"MDSimulation",self.xtc_fn.rsplit(".")[0])
        npy += Nuc + "_"
        if frame_A == "bond":
            npy += "b_"
        if frame_B == "chain_rotate":
            npy+= "cr"
        elif frame_B== "bond_rotate":
            npy+= "br"
        elif frame_B == "librations":
            npy += "libr"
        elif frame_B == "peptide_plane":
            npy+="pp"
        elif frame_B == "methylCC":
            npy+= "meCC"

        if os.path.exists(npy) and not reload:
            md_noopt = load_DIFRATE('HETs' + self.xtc_fn)
            mol = self.molecule
            nr = int(len(mol.mda_object.residues) / 71)
            mol.mda_object.residues.resids = np.atleast_2d(np.arange(219, 290)).repeat(nr, axis=0).reshape(nr * 71)
            self.md_data[self.xtc_fn] = md_noopt


        if calc:
            if tf or n_sparse>0:
                dump = False
            self.md_data[self.xtc_fn] = difrate.eval_fr.frames2data(self.molecule,
                                                                    n=n_sparse, tf=tf)  # to load_md
            #mol = self.molecule
            #mol.mda_object.residues.resids = np.array(
            #    [k * np.ones(int(len(mol.mda_object.residues) / 71)) for k in range(219, 290)]) \
            #    .T.reshape(len(mol.mda_object.residues))
            #mol.select_atoms(Nuc='ivla', segids='B')
            #md_in = difrate.Ct_fast.Ct2data(mol, n=-1,nt=10000)
            #md_in.detect.r_no_opt(15)
            #md_noopt = md_in.fit(save_input=False)
            #md_noopt.save(npy)
            #self.md_data[self.md_sel] = md_in
        if dump:
            pass
        return self.md_data[self.xtc_fn]


class HETs_NH(MolSystem):
    def compare(self,**kwargs):
        sum_experiments = len(self.nmr_raw_data.keys()) + len(self.md_data.keys())
        assert sum_experiments > 1, "You should provide more than 1 dataset to compare anything"
        exp = self.get_nmr_experiment("15N_data")
        Nuc = "N"
        seg = "B"
        kwargs.update(Nuc="N")
        kwargs.update(segids="B")
        kwargs.update(frame_A="bond")
        kwargs.update(frame_B="peptide_plane")
        mot = 3
        legend = []
        self.molecule.load_structure_from_file_path(0)
        self.molecule.select_atoms(Nuc=Nuc, segids=seg)
        exp.detect.r_auto(7 if not "n_det" in kwargs else kwargs["n_det"], inclS2=True, Normalization="MP")
        fit = exp.fit()
        residsMD = exp.label.astype(int)-74*2#range(min(exp.label.astype(int) - 74*2),max(exp.label.astype(int)-74*2))
        kwargs.update(resids=residsMD)
        fig,ax0, bx = fit.plot_rho(rho_index=range(4), errorbars=True)
        fig = plt.figure()
        ax0 = fig.add_subplot(511)
        a,b = fit.get_rho(rho_index=range(4))
        ax0.plot(a,b)
        ax = [fig.add_subplot(5,1,n+2) for n in range(4)]
        n_comp = kwargs.get("n_comp") if kwargs.get("n_comp") else 2
        w = .9/((n_comp if isinstance(n_comp,int) else len(n_comp))+1)
        for i,a in enumerate(ax):
            a.bar(exp.label,fit.R[:,i],width=w)
        targ = fit.sens.rhoz()[:4]
        targ[0, 84:] = 0
        legend.append("NMR")
        for j,i in enumerate(range(n_comp) if isinstance(n_comp,int) else n_comp):
            md = self.load_MD(sim=None if isinstance(n_comp,int) else i, **kwargs)
            md[mot].detect.r_target(target=targ, n=15)
            md[mot] = md[mot].fit()
            if j== 0:
                a,b = md[mot].get_rho(rho_index=range(3))
                ax0.plot(a,b,linestyle='dashed')
            legend.append(self.xtc_fn)
        x0 =exp.label.astype(int)#range(219,self.md_data[self.md_sel][mot].R.shape[0]+219)
        for i,sim in enumerate(self.md_data.keys()):
            for k, a in enumerate(ax):
                bx[k].scatter(x0, self.md_data[sim][mot].R[:, k],marker="x")
                #a.scatter(x0, self.md_data[sim][mot].R[:, k],marker="x")
                a.bar(x0+(i+1)*w, self.md_data[sim][mot].R[:, k],width=w)
                a.grid(True)


        '''print("Which simulation is closer?")
        for sim in self.md_data.keys():
            print(sim)
            for k in range(len(ax)):
                print("Detektor:",k)
                sim_data = self.md_data[sim][mot].R[:, k]
                fit_data = fit.R[:, k]
                s = []
                for num,lab in enumerate(exp.label):
                    d = sim_data[num]/fit_data[num] if sim_data[num] < fit_data[num] else fit_data[num] / sim_data[num]
                    s.append(d if d > 0 else 0)
                print(np.mean(s))
            print()'''
        plt.legend(legend, title="Simulations", bbox_to_anchor=(0, 3.5, 1, 0.0), mode="expand", ncol=4,
                       loc='lower left')
        plt.show()

if __name__=='__main__':
    MolSystem("HETs")