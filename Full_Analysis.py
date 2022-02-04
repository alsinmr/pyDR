from os import listdir, system, mkdir
from os.path import join, exists, abspath, dirname
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import MDAnalysis as MDA
from calcs import *
from SpeedTest import time_runtime, time
import sys
import pyDIFRATE as DR
from pyDIFRATE.Struct import vf_tools
from numba.cuda import is_available as cuda_available
import scipy
# setting default parameters for matplotlib plots

matplotlib.rcParams.update({"lines.linewidth": 1,
                            "axes.labelsize": 8,
                            "xtick.labelsize": 8,
                            "ytick.labelsize": 8,
                            "axes.titlesize": 10,
                            'font.size': 8})
TEXTBOX = {'facecolor': 'lightblue',
           'alpha': 0.5}




class KaiMarkov():
    def __init__(self, **kwargs):
        self.n = 200
        self.offset=0
        self.length = None
        self.full_traj = False
        self.exclude = []   #excluding specifig bonds by their label names
        self.labels = []
        self.full_chains = []
        self.plot_hydrogens_rama_3D = []  # get a better name for that, for ramachandran 3d plot
        self.n_dets = None  # number of detectors for analysis
        self.part = None  # the number of frames that will be in one fraction defined by length/n
        self.residues = []  # residues to examine, can be preset in kwargs
        self.pdb = None  # pdb file (path) that will fit to the selected simulation
        self.sel_sim = None  # filename of selected simulation
        self.sel_sim_name = None  # same just with removed xtc
        self.universe = None  # will be MDA.Universe object
        self.default_float = "float32"  # used for all kinds of calculations and saves some space compared to float 64
        self.save_float = "float16"  # saves space for saving issues, since md accuracy is only 0.01 it shouldn't matter
        self.v = False  # verbose, if activated, print
        self.full_dict = {}
        self.dir = dirname(__file__)
        if not exists(join(self.dir, "calced_data")):
            mkdir(join(self.dir, "calced_data"))
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.sim_dict = {'cts':None,
                         "S2s":None,
                         'dihedrals':None,
                         'residues':{}
                         }
        self.dihedral_atomgroups = []
        self.vector_atomgroups = []

        self.select_simulation(kwargs.get("simulation"))
        self.get_peptide_planes()
        self.get_methyl_groups()
        self.create_data_container()
        self.ct_x_axis = np.arange(1, self.length + 1, 1)

    def create_data_container(self):
        """
        This function creates the numpy arrays in full size for all necessary computations during the runtime of the
        program und linking them to the final dictionary
        :return: None
        """

        m = len(self.dihedral_atomgroups)
        n = self.length
        self.sim_dict["dihedrals"] = self.dihedrals = np.zeros((m, n), dtype= self.default_float)
        self.areas = np.zeros((m, n), dtype=self.default_float)
        self.hops = np.zeros((m, n), dtype=bool)
        self.hopability = np.zeros((m, n), dtype= self.default_float)

        m = len(self.vector_atomgroups)

        self.ct_vectors = np.zeros((m, n, 3), dtype=self.default_float)
        self.sim_dict["cts"] = self.cts = np.ones((m, n), dtype=self.default_float)
        self.sim_dict["S2s"] = self.S2s = np.zeros(m, dtype=self.default_float)

    def is_res_in_dict(self,resid:int):
        """
        check if the residue is already in the calculation dicitonary, if not, create a key and both a list for dihedrals
        and vor ct_vectors
        :param resid: integer, ID of the residue in the pdb file
        :return: resname (str) type+id
        """
        resname = self.universe.residues[resid-1].resname + str(resid)
        if not resname in self.sim_dict['residues'].keys():
            self.sim_dict['residues'][resname] = {"dihedrals":[], "ct_vecs":[]}
        return resname

    def add_dihedral_by_ids(self, atomlist : MDA.AtomGroup, name=""):
        if name in self.exclude:
            return
        #todo put this and the next function in one and add argument for choosing dihedral or vector
        assert len(atomlist)==4, "atomlist has to contain 4 atom ids, contains {}".format(len(atomlist))
        if isinstance(atomlist, list) or isinstance(atomlist,np.ndarray):
            #todo something is goofing around here, but i dont understand what
            atomlist = MDA.AtomGroup(self.universe.atoms[atomlist-1])
        resname = self.is_res_in_dict(atomlist[0].residue.resid)
        dih = {"atom_ids":[], "name":"", "id":None}
        dih["atom_ids"] = atomlist.ids
        dih["name"] = name
        self.sim_dict['residues'][resname]["dihedrals"].append(dih)
        self.dihedral_atomgroups.append(atomlist)
        dih["id"] = len(self.dihedral_atomgroups)-1


    def add_vector_for_ct_by_ids(self, atomlist : MDA.AtomGroup, name=""):
        if name in self.exclude:
            return
        assert len(atomlist)>=3 and len(atomlist)<=5, "atomlist has to contain 3 to 5 atom ids, contains {}".format(len(atomlist))
        if isinstance(atomlist, list) or isinstance(atomlist,np.ndarray):
            #todo something is goofing around here, but i dont understand what
            atomlist = MDA.AtomGroup(self.universe.atoms[atomlist-1])
        resname = self.is_res_in_dict(atomlist[0].residue.resid)
        vec = {"atom_ids":[], "name":"", "id":None}
        vec["atom_ids"] = atomlist.ids
        vec["name"] = name
        self.sim_dict['residues'][resname]["ct_vecs"].append(vec)
        self.vector_atomgroups.append(atomlist)
        vec["id"] = len(self.vector_atomgroups)-1

    def get_peptide_planes(self):
        """adding normal vectors for the peptide planes, defined by the position of N-C-O of the peptide (OC1 for
        terminal residue) """
        uni = self.universe  # just to shorten this
        # TODO this will just work for HETs right now, make it more general
        seg = 1 if len(uni.segments) == 3 else 2 if len(uni.segments) == 5 else 0
        # TODO add segment selection to select simulation function
        segment = uni.segments[seg]
        residues = [segment.residues[r] for r in self.residues] if len(self.residues) else segment.residues
        for i, res in enumerate(residues):
            atomgroup = []
            for look in ["N", "C", "O","OC1"]:
                for atom in res.atoms:
                    if atom.name==look:
                        atomgroup.append(atom)
            assert len(atomgroup) == 3, "length of this atomgroup should be 3, but is {}".format(len(atomgroup))
            self.add_vector_for_ct_by_ids(MDA.AtomGroup(atomgroup), "peptide_plane")

        for i, res in enumerate(residues):
            atomgroup = []
            for look in ["N", "C","O","OC1","H","H1"]:
                for atom in res.atoms:
                    if atom.name==look:
                        atomgroup.append(atom)
            assert len(atomgroup) == 4, "length of this atomgroup should be 4, but is {}".format(len(atomgroup))
            self.add_vector_for_ct_by_ids(MDA.AtomGroup(atomgroup), "NH-Bond lib.")

    def get_methyl_groups(self):
        #return an MDA Atomgroup from a list of Atoms
        make_AG = lambda alist: MDA.AtomGroup([this[_] for _ in alist])
        assert self.sel_sim and self.pdb, "You have to select a PDB file AND a simulation before "
        if self.universe is None:
            self.universe = MDA.Universe(self.pdb, self.sel_sim)
        uni = self.universe  # just to shorten this
        # TODO this will just work for HETs right now, make it more general
        seg = 1 if len(uni.segments) == 3 else 2 if len(uni.segments) == 5 else 0
        # TODO add segment selection to select simulation function
        segment = uni.segments[seg]
        residues = [segment.residues[r] for r in self.residues] if len(self.residues) else segment.residues
        dix = self.full_dict
        for i, res in enumerate(residues):
            methyls = search_methyl_groups(res, v=self.v)
            if len(methyls):
                group_label = "{}_{}".format(res.resname, res.resnum + 147)
                this = dix[group_label] = {}
                this["ct_indices"] = []
                this["ct_labels"] = []
                print(i, res.resname, res.resnum + 147, "contains", len(methyls), "methyl groups")
                chain_for_3d_plot = dix[group_label]["chain"] = ["C", "CA", "N", "CA", "CB"]
                this["N"] = res.atoms[0]
                this["H1"] = methyls[0][0]  # this is only useful if you have only one H
                this["H12"] = methyls[0][1]
                this["H13"] = methyls[0][2]
                this["C"] = methyls[0][-1]
                this["CA"] = methyls[0][-2]
                this["CB"] = methyls[0][-3]
                self.add_vector_for_ct_by_ids(make_AG(["CB", "N", "CA", "C"]),r'$\chi_{1,lib.}$' if "ALA" not in res.resname else r'$CH_{3,lib.}$')
                # todo add methionin
                if "ILE" in res.resname:
                    for _ in ["CG2", "CB", "CG1", "CD"]:
                        chain_for_3d_plot.append(_)
                    this["CG2"] = methyls[0][-4]
                    this["CG1"] = methyls[1][4]
                    this["CD"] = methyls[1][3]
                    this["H2"] = methyls[1][0]
                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG1"]), "chi1")
                    self.add_dihedral_by_ids(make_AG(["CA", "CB", "CG1", "CD"]), "chi2")
                    self.add_vector_for_ct_by_ids(make_AG(["CG1", "C", "CB", "CA"]), r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CG1", "CA", "CB", "CG2"]), r'$\chi_{2,lib.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CD", "CB", "CG1", "CG2"]), r'$\chi_{2,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CB", "CG2", "CG1"]), r'$CH_{3,rot.}^1$')
                    self.add_vector_for_ct_by_ids(make_AG(["H2", "CG1", "CD", "CB"]), r'$CH_{3,rot.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CG2"]),"met-to-plane")
                    self.add_vector_for_ct_by_ids(make_AG(["H2", "N", "CA", "C", "CD"]), "met-to-plane_2")
                    self.add_vector_for_ct_by_ids(make_AG(["CD","N","CA","C","CG1"]),"C-C-2-plane")

                elif "LEU" in res.resname:
                    for _ in ["CG", "CD1", "CG", "CD2"]:
                        chain_for_3d_plot.append(_)
                    this["CG"] = methyls[0][4]
                    this["CD1"] = methyls[0][3]
                    this["CD2"] = methyls[1][3]
                    this["H2"] = methyls[1][0]
                    self.add_vector_for_ct_by_ids(make_AG(["CG", "CA", "CB", "C"]),r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CD1", "CA", "CG", "CB"]),r'$\chi_{2,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CG", "CD1", "CD2"]),r'$CH_{3,rot.}^1$')
                    self.add_vector_for_ct_by_ids(make_AG(["H2", "CG", "CD2", "CD1"]),r'$CH_{3,rot.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CD1"]),"met-to-plane")
                    self.add_vector_for_ct_by_ids(make_AG(["H2","N","CA","C","CD2"]),"met-to-plane_2")

                    self.add_vector_for_ct_by_ids(make_AG(["CD2","N","CA","C","CG"]),"C-C-2-plane")

                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG"]),"chi1")
                    self.add_dihedral_by_ids(make_AG(["CA","CB","CG","CD2"]),"chi2")
                elif "VAL" in res.resname:
                    for _ in ["CG1", "CB", "CG2"]:
                        chain_for_3d_plot.append(_)
                    this["CG1"] = methyls[0][-4]  # in "VAL" this is CG1
                    this["CG2"] = methyls[1][3]  # in "VAL this is CG2
                    this["H2"] = methyls[1][0]
                    self.add_vector_for_ct_by_ids(make_AG(["CG1","C","CB","CA"]),r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CG2", "CG1", "CB", "CA"]), r'$CH_{3,lib.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CB", "CG1", "CG2"]), r'$CH_{3,rot.}^1$')
                    self.add_vector_for_ct_by_ids(make_AG(["H2", "CB", "CG2", "CG1"]), r'$CH_{3,rot.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CG1"]), "met-to-plane")
                    self.add_vector_for_ct_by_ids(make_AG(["H2","N","CA","C","CG2"]), "met-to-plane_2")
                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG2"]),"chi1")
                elif "THR" in res.resname:
                    this["CG2"] = methyls[0][3]
                    chain_for_3d_plot.append("CG2")
                    self.add_vector_for_ct_by_ids(make_AG(["CG2", "CA", "CB", "C"]),r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CA", "CG2", "CB"]),r'$CH_{3,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CG2"]),"met-to-plane")
                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG2"]),"chi1")
                elif "ALA" in res.resname:
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CA", "CB", "C"]),r'$CH_{3,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CB"]),"met-to-plane")
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "H12", "CB", "CA"]),r'$C-H_{lib.}$')
                this["labels"] = []
                for j, m in enumerate(methyls):
                    met_label = group_label + ("_B" if j else ("_A" if len(methyls) == 2 else ""))
                    this["labels"].append(met_label)
                    this[met_label] = {}
                    ###sdix = this[met_label]  # subdictionary
                    ###sdix["index"] = len(self.hydrogens_dihedral)
                    ###sdix["carbon"] = m[3]
                    group = MDA.AtomGroup(m)
                    self.full_chains.append(group)
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 0]])
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 1]])
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 2]])
                    # One group appended for every hydrogen
                    # it is a little weird but still this was the fastest way to calculate the dih for this
                    ###self.hydrogens_dihedral.append(
                    ###    group[[0, 3, 4, 5]])  # presetting this saves much time for calculations
                    ###self.hydrogens_dihedral.append(group[[1, 3, 4, 5]])
                    ###self.hydrogens_dihedral.append(group[[2, 3, 4, 5]])
                    self.add_dihedral_by_ids(group[[0,3,4,5]],"CH3_{}".format(j+1))

        ###assert len(self.hydrogens_dihedral), "No methylgroups detected"
        ###print("Total:", int(len(self.hydrogens_dihedral) / 3), "methylgroups detected")
        if self.v:
            for key in dix.keys():
                print(key)
                print(dix[key])

    def select_simulation(self, number=None):
        """
        Selecting a simulation from the folders xtcs and pdbs. If no number is provided, a list of available xtc-files
        is printed to choose a simulation. The fitting pdb file is selected automatically
        :param number: int
        :return: nothing
        """
        xtcdir = join(self.dir, "xtcs")
        pdbdir = join(self.dir, "pdbs")
        if not exists(xtcdir) or not exists(pdbdir):
            mkdir(xtcdir)
            mkdir(pdbdir)
            assert 0, "folders for xtcs and pdbs were created, plz fill with trajectories and pdbfiles"

        if not self.sel_sim:
            simulations = [f for f in listdir(xtcdir) if f.endswith(".xtc")]
            assert len(simulations), "please copy (or link) some simulations in your xtcs/ folder"
            assert len(listdir(pdbdir)), "of course you also need a pdb file (in pdbs/)"
            if number is None:
                print("Available Simulations:")
                for i, sim in enumerate(simulations):
                    print(i, sim)
                i = int(input("Choosen: "))
            else:
                i = number
            try:
                self.sel_sim = join(xtcdir, simulations[i])
                self.sel_sim_name = simulations[i].split(".")[0]
            except:
                print("not a valid simulation")
                self.select_simulation()
        else:
            self.sel_sim_name = self.sel_sim.split(".")[0].split("/")[-1]
        for pdb in listdir(pdbdir):
            f = join(pdbdir, pdb)
            print(f)
            try:
                self.universe = MDA.Universe(f, self.sel_sim)
                self.pdb = f
                break
            except:
                self.universe = None

        assert self.universe, "Couldnt find a fitting pdb for this simulation, please copy one in the pdb folder"
        print("Simulation", self.sel_sim_name, "with timestep", self.universe.trajectory.dt, "and", self.pdb,
              "selected")
        self.dt = self.universe.trajectory.dt * 10 ** (-12)  # 5 ps
        if self.length == None:
            self.length = len(self.universe.trajectory)
            self.full_traj = True
        # todo put segment selection here if more than one segment is available

    def load(self):
        """
        Loading previously calculated dihedral angles and correlation functions. A bool array for each group is generated
        to determine which angle/bond is already calculated in case one was just adding a new bond to the dictionary.
        If the length of the trajectory extends the length of the stored arrays, the whole calculation has to run again

        While the data is usually saved as float16 it will automatically be recorvered in the default calculation
        datatye (float32)
        :return: Nothing
        """
        dihedral_calc_indices = np.ones(len(self.dihedral_atomgroups)).astype(bool)
        ct_calc_indices = np.ones(len(self.vector_atomgroups)).astype(bool)
        filename = join("calced_data", self.sel_sim_name + ".npy")
        if exists(filename):
            loaded = np.load(filename, allow_pickle=True).item()
            for key in self.sim_dict['residues'].keys():
                if not key in loaded:
                    continue
                for dih in self.sim_dict['residues'][key]['dihedrals']:
                    for dih2 in loaded[key]['dihedrals']:
                        if (dih['atom_ids'] == dih2['atom_ids']).sum() == len(dih['atom_ids']) \
                                and dih2['values'].shape[0] >= self.length:
                            self.sim_dict['dihedrals'][dih['id']] = dih2['values'][:self.length,]
                            dihedral_calc_indices[dih['id']]=0
                for ct in self.sim_dict['residues'][key]['ct_vecs']:
                    for ct2 in loaded[key]["ct_vecs"]:
                        if len(ct['atom_ids']) == len(ct2['atom_ids']):
                            if (ct['atom_ids'] == ct2['atom_ids']).sum() == len(ct['atom_ids']) \
                                    and ct2['values'].shape[0] >= self.length:
                                self.sim_dict['cts'][ct['id']] = ct2['values'][:self.length,]
                                self.sim_dict['S2s'][ct['id']] = ct2['S2']
                                ct_calc_indices[ct['id']] = 0
        return dihedral_calc_indices, ct_calc_indices

    def save(self, dtype="float16"):
        """
        saving the results of dihedral calculation, correlation functions and S2 in a numpy dictionary inside the folder
        calced_data as 'simulation_name'.npy
        Note that by default the data is saved with reduced accuracy (float16) to save disc space
        :return:
        """
        #assert 0, "change before save again!"
        #rearrange the dictionary, just save the 'residues' part and remove the index, put instead the values
        #directly to the thing

        #todo add sparse values to dict
        for res in self.sim_dict['residues'].keys():
            for dih in self.sim_dict['residues'][res]['dihedrals']:
                dih["values"] = self.sim_dict['dihedrals'][dih['id']].astype(dtype)
            for ct in self.sim_dict['residues'][res]['ct_vecs']:
                ct["values"] = self.sim_dict['cts'][ct['id']].astype(dtype)
                ct["S2"] = self.sim_dict['S2s'][ct['id']].astype(dtype)

        self.sim_dict["cts"] = self.sim_dict["cts"].astype(dtype)
        self.sim_dict["dihedrals"] = self.sim_dict["dihedrals"].astype(dtype)
        self.sim_dict["S2s"] = self.sim_dict["S2s"].astype(dtype)
        np.save(join("calced_data",self.sel_sim_name+".npy"), self.sim_dict['residues'])

    def calc(self, **kwargs):
        """
        Iterating over the full trajectory and calculating dihedral angles and correlation functions for every selected
        bond in every selected residue
        :param kwargs:
        :param force_calculation: bool, force a full calculation of all dihedrals and correlation funcitons
        :return:
        """
        traj = self.universe.trajectory
        assert self.length <= len(traj), "Length {} exceeding length of trajectory ({}), set length to {}" \
                                         "".format(self.length, len(traj), len(traj))
        dihedral_calc, ct_calc = self.load()
        if kwargs.get("force_calculation"):
            dihedral_calc[:] = 1
            ct_calc[:] = 1
        print("Dihedrals to calculate: {}\nCorrelation functions to calculate: {}".format(dihedral_calc.sum(), ct_calc.sum()))

        if dihedral_calc.sum() or ct_calc.sum():
            if exists("long_traj_vecs.npy") and exists("long_traj_dihedrals.npy"):
                # because of some issues with ct calculation (for long trajectories with many residues)
                # i decided to save the vectors after loading them
                # if calculation fails, they will just be reloaded the next time
                # if it succeeds, we delete the files in the end
                self.dihedrals = np.load("long_traj_dihedrals.npy").astype(self.default_float)
                self.ct_vectors = np.load("long_traj_vecs.npy").astype(self.default_float)
            else:
                for i,_ in enumerate(range(self.offset,self.offset+self.length)):
                    if _ % 1000==0: print(_)
                    traj[_]
                    for j, group in enumerate(self.dihedral_atomgroups):
                        #todo put all atoms in ONE group and make fastest dihedral run in parallel for all atomgroups at the
                        #todo same time, check the speedup!
                        if dihedral_calc[j]:
                            self.dihedrals[j,i] = fastest_dihedral(group.positions)
                    for j, group in enumerate(self.vector_atomgroups):
                        if ct_calc[j]:
                            if len(group)>3:
                                pos_xyz_o(self.ct_vectors[j, i], *group.positions)
                            else:
                                self.ct_vectors[j,i] = get_peptide_plane_normal(group.positions)

                np.save("long_traj_vecs.npy",self.ct_vectors.astype("float16"))
                np.save("long_traj_dihedrals.npy", self.dihedrals.astype("float16"))
            if ct_calc.sum():
                if cuda_available():
                    count =ct_calc.sum()
                    chunk=1
                    while count > 50:
                        chunk<<=1
                        count>>=1
                    count+=1

                    for i in range(chunk+1):
                        ct_calc[:]=0
                        ct_calc[i*count:(i+1)*count] = 1
                        if ct_calc.sum():
                            calc_CT_on_cuda(self.cts,self.S2s, self.ct_vectors, ct_calc,
                              kwargs.get("sparse") if kwargs.get("sparse") is not None else 1)
                else:
                    get_ct_S2(self.cts, self.S2s, self.ct_vectors, ct_calc,
                          kwargs.get("sparse") if kwargs.get("sparse") is not None else 1)

            if self.full_traj and len(self.residues) == 0 and kwargs.get("sparse") == 0:
                self.save()

    def construct_full_ct_from_states(self, Markov_states, avg_vecs, axis, **kwargs):
        """
        Constructing an ct from Markov states with average vectors of the bond and plot them on an axis
        :param Markov_states: np.ndarray with integers to represent a markov state
        :param avg_vecs: np.ndarray with shape n x 3
        :param axis: an axis of matplotlib to plot the resulting correlation funciton
        :param kwargs: sparse value (integer) with default 50
        :return: nothing
        """
        ct = np.zeros((1,Markov_states.shape[0]),dtype="float32")
        vecs = np.zeros((1,Markov_states.shape[0],3),dtype="float32")
        for i in range(Markov_states.shape[0]):
            vecs[0, i] = avg_vecs[Markov_states[i]]

        sparse = kwargs.get("sparse") if kwargs.get("sparse") else 50
        calc_CT_on_cuda(ct, np.zeros(1), vecs, np.ones(1, dtype=bool), sparse=sparse)

        ticks = np.array([10, 100, 1000, 10000, 100000, 1000000]).astype(int)
        ticks //= int(self.dt*(10**12))

        axis.semilogx(self.ct_x_axis, ct[0], color="green", linestyle="dotted")
        axis.set_ylim(0,1.05)
        axis.set_xticks(ticks)

        axis.set_yticks([0, .25, .5, .75, 1])
        axis.set_xticklabels(["10 ps", "100 ps", "1 ns", "10 ns", "100 ns", "1 µs"], rotation=30)

    def do_markov(self, num=0):
        '''working markov example for chi1 and chi2 bond, examining the CC-bond of the second methyl group and
        determining the correlation funciton from that
        this funciton is very specific for simulation HETs-MET-4pw and is not by default transferable on other
        simulations
        '''
        vecs = np.load("long_traj_vecs.npy").astype(self.default_float)
        res = []
        chi1 = []
        chi2 = []
        cc_vecs = []
        '''searching the simulation dictionary for residues Isoleucin and Leucin to get the ID's of chi1 and chi2
        dihedrals'''
        for key in self.sim_dict["residues"].keys():
            if "ILE" in key or "LEU" in key:
                res.append(key)
                for dih in self.sim_dict['residues'][key]['dihedrals']:
                    if "chi1" in dih['name']:
                        chi1.append(dih['id'])
                    if "chi2" in dih['name']:
                        chi2.append(dih['id'])
                    #if "CH3_1" in dih['name']:
                    #    ch3_1.append(dih['id'])
                    #if "CH3_2" in dih['name']:
                    #    ch3_2.append(dih['id'])
                for vec in self.sim_dict["residues"][key]['ct_vecs']:
                    if "C-C-2-plane" in vec['name']:
                        cc_vecs.append(vec['id'])

        cc_vecs = np.array(cc_vecs).astype(int)
        vecs = vecs[cc_vecs]
        chi1 = np.array(chi1).astype(int)
        chi2 = np.array(chi2).astype(int)


        chi1_states = np.zeros((9,self.dihedrals.shape[1]))
        chi2_states = np.zeros((9,self.dihedrals.shape[1]))
        states = np.zeros((9, self.dihedrals.shape[1])).astype(int)
        chi1_states[:, :] += ((self.dihedrals[chi1, :] >= 0) == (self.dihedrals[chi1, :] < 120)).astype('uint8')
        chi1_states[:, :] += ((self.dihedrals[chi1, :] < 0) == (self.dihedrals[chi1, :] >= -120)).astype('uint8') * 2
        chi2_states[:, :] += ((self.dihedrals[chi2, :] >= 0) == (self.dihedrals[chi2, :] < 120)).astype('uint8')
        chi2_states[:, :] += ((self.dihedrals[chi2, :] < 0) == (self.dihedrals[chi2, :] >= -120)).astype('uint8') * 2

        for m in [0, 1, 2]:
            for n in [0, 1, 2]:
                states[(chi1_states == m) & (chi2_states == n)] = 3 * n + m


        fig = plt.figure()
        ax = [fig.add_subplot(2,3,i+1) for i in range(5)]
        plt.suptitle(res[num])
        ax[0].hist2d((self.dihedrals[chi1[num]] + 240) % 360,
                     (self.dihedrals[chi2[num]] + 240) % 360, range=[[0, 360], [0, 360]], bins=[180, 180], cmin=0.1)
        ax[4].scatter(((self.dihedrals[chi1[num]]+240) % 360)[::100],
                      ((self.dihedrals[chi2[num]]+240) % 360)[::100], s=0.25, c=states[num, :][::100])
        x = [60, 300, 180, 60, 300, 180, 60, 300, 180]  # this is a little goofy, but only this way it plots on the
        y = [60, 60, 60, 300, 300, 300, 180, 180, 180]  # right spots
        avg_vecs = []
        avg_vecs_B = []  # separate list to avoid errors resulting from reduced matrices
        for i in range(9):
            '''averaging all 9 vectors in their specific state and store them for calculation of the P2 for C(t) later
            vectors should be normalized'''
            avg_vec = vecs[num, states[num, :] == i]
            avg_vec = np.average(avg_vec, axis=0)
            avg_vec /= np.linalg.norm(avg_vec)
            if not np.isnan(avg_vec[0]):
                avg_vecs.append(avg_vec)
            avg_vecs_B.append(avg_vec)
            ax[4].text(x[i], y[i], "{:.2f}\n{:.2f}\n{:.2f}".format(*avg_vec), ha="center", va="center")

        avg_vecs = np.array(avg_vecs)
        transition_matrix = np.zeros((9, 9))
        to_delete = []  # if a state is not populated, the dimension of the matrix will be removed after the creation of
                        # the transition matrix
        for i in range(9):
            for j in range(9):
                '''here we calculate the probability to go from state i to state j to create the transition matrix'''
                if (states[num,] == i).sum():
                    transition_matrix[j, i] = prob = ((states[num, :-1] == i) & (states[num, 1:] == j)).sum() / \
                                                     (states[num, ] == i).sum()
                    if i == j:
                        if not np.isnan(prob):
                            '''plot the probability to stay in the same state inside of the Ramachandran plot'''
                            ax[0].text(x[i], y[i], "{:.2f}".format(prob), ha="center", color="black", bbox=TEXTBOX)
                    else:
                        '''drawing an arrow between the states with a thickness to represent the transition probability
                        between these states'''
                        if not np.isnan(prob):
                            ax[0].arrow(x[i]+np.random.random()*25, y[i]+np.random.random()*25, x[j]-x[i],y[j]-y[i],
                                        alpha=prob, width=prob/10)
                    prob *= 100

                else:
                    if i == j:
                        to_delete.append(i)

        for index in to_delete[::-1]:
            transition_matrix = np.delete(transition_matrix, index, 1)
            transition_matrix = np.delete(transition_matrix, index, 0)

        transition_matrix /= transition_matrix.sum(axis=0)  # normalization
        ax[1].imshow(transition_matrix, vmin=0, vmax=1)
        ax[3].semilogx(self.ct_x_axis,self.cts[cc_vecs[num]],color="black")
        P2matrix = np.zeros(transition_matrix.shape)
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[0]):
                P2matrix[j, i] = P2(np.dot(avg_vecs[i], avg_vecs[j]))#

        pop = []
        for l in range(9):
            if((states[num, :] == l).sum()) / self.length:
                pop.append(((states[num, :] == l).sum()) / self.length)
        pop = np.array(pop)
        print(pop, pop.sum())

        tp = np.zeros(self.length)

        for l in range(transition_matrix.shape[0]):
            init = np.zeros(transition_matrix.shape[0])
            init[l]=1
            ic = np.ones(self.length)
            for i in range(1,self.length):
                if i % 100000 == 0: print(i, init.sum())
                init = transition_matrix @ init
                ic[i] = (P2matrix[l] * init).sum()
            tp += ic * pop[l]

        ax[3].semilogx(self.ct_x_axis, tp, color="red")
        self.construct_full_ct_from_states(states[num], avg_vecs_B, ax[3])

        for col in range(transition_matrix.shape[0]):
            if (transition_matrix[col, :] == 1).sum():
                transition_matrix[col, transition_matrix[col, :] == 1] = 1 - (transition_matrix.shape[0] - 1) * .001
                transition_matrix[col, transition_matrix[col, :] == 0] = 0.001

        exchange_matrix = scipy.linalg.logm(transition_matrix) / 5  #or *5?
        try:
            ax[2].imshow(exchange_matrix.T)
        except:
            print("Exchange Matrix for {} could not be plotted probably because of irrational values".format(res[num]))

        # just some plotting refinements
        for a in [ax[0], ax[4]]:
            a.set_xlabel(r'$\chi_1$')
            a.set_ylabel(r'$\chi_2$')
            a.set_xticks([60, 180, 300])
            a.set_yticks([60, 180, 300])
            a.set_xlim(0, 360)
            a.set_ylim(0, 360)
            a.tick_params(axis="y", rotation=90)
        for a in [ax[1],ax[2]]:
            a.set_xticks(range(transition_matrix.shape[0]))
            a.set_yticks(range(transition_matrix.shape[0]))
            a.set_xticklabels([])
            a.set_yticklabels([])

        fig.subplots_adjust(wspace=0.22, hspace=0.2, right=0.99, top=0.95, bottom=0.075, left=0.075)
        plt.savefig("{}_C-C_bond.pdf".format(res[num]))

    def do_markov_3dihedrals(self, num:int):
        """
        basically the same funciton as do_markov, just here we take a third dihedral, the one of the methylgroup itself
        in account
        :param num: integer
        :return:
        """
        vecs = np.load("long_traj_vecs.npy")[:, :self.length].astype(self.default_float)
        res = []
        chi1 = []
        chi2 = []
        ch3_2 = []
        cc_vecs = []
        for key in self.sim_dict["residues"].keys():
            if "ILE" in key or "LEU" in key:
                res.append(key)
                for dih in self.sim_dict['residues'][key]['dihedrals']:
                    if "chi1" in dih['name']:
                        chi1.append(dih['id'])
                    if "chi2" in dih['name']:
                        chi2.append(dih['id'])
                    if "CH3_2" in dih['name']:
                        ch3_2.append(dih['id'])
                for vec in self.sim_dict["residues"][key]['ct_vecs']:
                    if "met-to-plane_2" in vec['name']:
                        cc_vecs.append(vec['id'])
        cc_vecs = np.array(cc_vecs).astype(int)
        vecs = vecs[cc_vecs]
        chi1 = np.array(chi1).astype(int)
        chi2 = np.array(chi2).astype(int)
        ch3_2 = np.array(ch3_2).astype(int)


        '''determining in which state each chi-bond or the methyl group is inside, depending on the dihedral angle'''
        chi1_states = np.zeros((9, self.dihedrals.shape[1]))
        chi2_states = np.zeros((9, self.dihedrals.shape[1]))
        ch3_2_states = np.zeros((9, self.dihedrals.shape[1]))

        chi1_states[:, :] += ((self.dihedrals[chi1, :] >= 0) == (self.dihedrals[chi1, :] < 120)).astype('uint8')
        chi1_states[:, :] += ((self.dihedrals[chi1, :] < 0) == (self.dihedrals[chi1, :] >= -120)).astype('uint8') * 2
        chi2_states[:, :] += ((self.dihedrals[chi2, :] >= 0) == (self.dihedrals[chi2, :] < 120)).astype('uint8')
        chi2_states[:, :] += ((self.dihedrals[chi2, :] < 0) == (self.dihedrals[chi2, :] >= -120)).astype('uint8') * 2
        ch3_2_states[:, :] += ((self.dihedrals[ch3_2, :] >= 0) == (self.dihedrals[ch3_2, :] < 120)).astype('uint8')
        ch3_2_states[:, :] += ((self.dihedrals[ch3_2, :] < 0) == (self.dihedrals[ch3_2, :] >= -120)).astype('uint8') *2

        states = np.zeros((9, self.dihedrals.shape[1])).astype(int)
        for k in [0, 1, 2]:
            for m in [0, 1, 2]:
                for n in [0, 1, 2]:
                    states[(chi1_states == m) & (chi2_states == n) &( ch3_2_states == k)] = 9 * k + 3 * n + m
                    # this calculation will result in 1 of 27 states for each timepoint and residue

        fig = plt.figure()
        ax = [fig.add_subplot(2, 3, i + 1) for i in range(5)]
        fig.suptitle(res[num])
        ax[0].hist2d((self.dihedrals[chi1[num]]+240) % 360,
                     (self.dihedrals[chi2[num]]+240) % 360, range=[[0, 360],[0, 360]],bins=[180, 180],cmin=0.1)
        ax[4].scatter(((self.dihedrals[chi1[num]]+240) % 360)[::100],
                      ((self.dihedrals[chi2[num]]+240) % 360)[::100], s=0.25, c=states[num, :][::100])
        avg_vecs_indices = []
        avg_vecs = []
        avg_vecs_B = []
        for i in range(27):
            avg_vec = vecs[num,(states[num,:]==i)]
            avg_vec = np.average(avg_vec,axis=0)
            avg_vec/=np.linalg.norm(avg_vec)
            if not np.isnan(avg_vec[0]):
                avg_vecs.append(avg_vec)
                avg_vecs_indices.append(i)
            avg_vecs_B.append(avg_vec)

        avg_vecs = np.array(avg_vecs)
        transition_matrix = np.zeros((27, 27))
        to_delete=[]
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[0]):
                if (states[num,]==i).sum():
                    transition_matrix[j, i] = ((states[num, :-1] == i) & (states[num, 1:] == j)).sum() / \
                                              (states[num, ] == i).sum()
                else:
                    if i==j:
                        to_delete.append(i)

        for index in to_delete[::-1]:
            transition_matrix = np.delete(transition_matrix, index, 1)
            transition_matrix = np.delete(transition_matrix, index, 0)

        transition_matrix /= transition_matrix.sum(axis=0)
        ax[1].imshow(transition_matrix, vmin=0, vmax=1)
        ax[3].semilogx(self.ct_x_axis,self.cts[cc_vecs[num]],color="black")
        P2matrix = np.zeros(transition_matrix.shape)
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[0]):
                P2matrix[j, i] = P2(np.dot(avg_vecs[i], avg_vecs[j]))

        pop = []
        for i in range(27):
            if((states[num, :] == i).sum()) / self.length:
                pop.append(((states[num, :] == i).sum()) / self.length)
        pop = np.array(pop)

        timepoints = np.zeros(self.length)
        for j in range(transition_matrix.shape[0]):
            probability_vector = np.zeros(transition_matrix.shape[0])
            probability_vector[j]=1
            ic = np.ones(self.length)
            for i in range(1,self.length):
                if i % 100000 == 0: print(i, probability_vector.sum())
                probability_vector = transition_matrix @ probability_vector
                ic[i] = (P2matrix[j] * probability_vector).sum()
            timepoints += ic * pop[j]

        ax[3].semilogx(self.ct_x_axis, timepoints, color="red")
        self.construct_full_ct_from_states(states[num],avg_vecs_B,ax[3])

        for col in range(transition_matrix.shape[0]):
            if (transition_matrix[col, :] == 1).sum():
                transition_matrix[col, transition_matrix[col, :] == 1] = 1 - (transition_matrix.shape[0] - 1) * .001
                transition_matrix[col, transition_matrix[col, :] == 0] = 0.001

        exchange_matrix = scipy.linalg.logm(transition_matrix) / 5  #or *5?
        try:
            ax[2].imshow(exchange_matrix.T)#
        except:
            print("Exchange matrix for {} could not be plotted".format(res[num]))
        for a in [ax[0],ax[4]]:
            a.set_xlabel(r'$\chi_1$')
            a.set_ylabel(r'$\chi_2$')
            a.set_xticks([60, 180, 300])
            a.set_yticks([60, 180, 300])
            a.set_xlim(0, 360)
            a.set_ylim(0, 360)
            a.tick_params(axis="y",rotation=90)
        for a in [ax[1],ax[2]]:
            a.set_xticks(range(transition_matrix.shape[0]))
            a.set_yticks(range(transition_matrix.shape[0]))
            a.set_xticklabels([])
            a.set_yticklabels([])
        fig.subplots_adjust(wspace=0.22, hspace=0.2, right=0.99, top=0.95, bottom=0.075, left=0.075)
        plt.savefig("{}_C-H_bond.pdf".format(res[num]))

    def plot_backbone_dynamics(self, a=0, calc_type=None):
        for key in self.sim_dict['residues']:
            print(key)
            for vec in self.sim_dict['residues'][key]['ct_vecs']:
                if vec['name'] == "peptide_plane":
                    last_id = vec['id']
        cts = self.cts[:last_id+1]
        #cts2 = self.cts[last_id+1:2*last_id+2]
        #cts*=cts2
        n_dets = self.n_dets if self.n_dets else 5
        page_2 = plt.figure()

        page_2.set_size_inches(8.3*3, 11.7 / 8 * n_dets)  # reduce page size for less cts


        nmr = DR.io.load_NMR('HETs_15N.txt')
        nmr.S2 *= (.103 / .102) ** 6

        n = 5
        nmr.detect.r_auto3(n-1 , inclS2=True, Normalization='MP')
        nmrfit = nmr.fit()
        nmrfit.del_data_pt(range(nmrfit.R.shape[0] - 5, nmrfit.R.shape[0]))
        target = nmrfit.sens.rhoz()[:4]
        target[0, 95:] = 0



        D = DR.data()
        D.load(Ct={'Ct': cts #if a ==0 else cts2
            , 't': np.linspace(0, self.length // 1000 * self.universe.trajectory.dt, self.length)})

        remove_S2 = 1
        D.detect.r_target(target, n=12)
        fit = D.fit()

        ax0 = page_2.add_subplot(n_dets, 1, 1)
        ax0.plot(D.sens.z(), nmrfit.sens.rhoz().T)
        ax0.set_xlim(-14, -3)
        ax0.set_title("Detector Sensitivities")
        ax0.set_ylabel(r'$(1-S^2)$')
        ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
        ax0.set_yticks([0, 1])
        ax = [page_2.add_subplot(n_dets, 1, 2 + i) for i in range(n_dets-2)]
        fit.label = (np.arange(71)+147+72).astype(int)

        for n,a in enumerate(ax):
            if n < 3:
                for i in range(len(nmrfit.label)):
                    a.bar(int(nmrfit.label[i]), nmrfit.R[i, n], color=plt.get_cmap("tab10")(n))
                    a.errorbar(int(nmrfit.label[i]), nmrfit.R[i,n],yerr=nmrfit.R_std[i,n], color = "black",capsize=5)
                a.plot(fit.label,fit.R[:,n],color="black")
                a.set_xlim(222,280)
            if n <2:
                a.set_xticklabels([])

        chi_hoch_zwei = 0
        for n in range(3):
            for i,nmrlab in enumerate(nmrfit.label):
                for j,mdlab in enumerate(fit.label):
                    if nmrlab==mdlab:
                        print(nmrlab,mdlab)
                        if calc_type=="mean":
                            chi = ((nmrfit.R[i, n] - fit.R[j, n]) ** 2) / (np.mean(nmrfit.R_std[:, n]) ** 2)
                        elif calc_type=="median":
                            chi =((nmrfit.R[i,n] - fit.R[j,n])**2)/(np.median(nmrfit.R_std[:,n])**2)
                        elif calc_type==None:
                            chi = ((nmrfit.R[i, n] - fit.R[j, n]) ** 2) / (nmrfit.R_std[i, n] ** 2)
                        chi_hoch_zwei += chi
                        ax[n].text(mdlab, nmrfit.R[i,n]*1.5+.05*(i%2), str(int(chi)),ha="center")

        print(chi_hoch_zwei)
        page_2.suptitle("Backbone - Chi^2:"+str(int(chi_hoch_zwei))+" - " + self.sel_sim_name + " {} R_std".format(calc_type))
        page_2.tight_layout()
        page_2.subplots_adjust(hspace=.1)
        if a==0:
            page_2.savefig("Peptide-Plane.pdf")
        else:
            page_2.savefig("NH-bond.pdf")



    def plot_single_res(self, label:str, plot_ct = False):
        def tc_str(z):
            unit = 's'
            if z <= -12: z, unit = z + 15, 'fs'
            if z <= -9: z, unit = z + 12, 'ps'
            if z <= -6: z, unit = z + 9, 'ns'
            if z <= -3: z, unit = z + 6, r'$\mu$s'
            if z <= 0: z, unit = z + 3, 'ms'
            tc = np.round(10 ** z, -1 if z >= 2 else (0 if z >= 1 else 1))
            return '{:2} '.format(tc if tc < 10 else int(tc)) + unit
        if plot_ct:
            fig = plt.figure()
            fig.set_size_inches(9, 8)

            ax = fig.add_subplot(211)
            ax.semilogx(np.array([self.cts[vec['id']] for vec in self.sim_dict['residues'][label]['ct_vecs']]).T)
            ax.legend([ct['name'] for ct in self.sim_dict['residues'][label]['ct_vecs']])
            ax.set_ylim(-.1,1.05)
            bx = fig.add_subplot(212)
            bx.plot(np.array([np.sort(self.dihedrals[dih['id']]) for dih in self.sim_dict['residues'][label]['dihedrals']]).T)
            bx.legend([dih['name'] for dih in self.sim_dict['residues'][label]['dihedrals']])

            fig.suptitle(label + " - " + self.sel_sim_name)

        if not label in self.sim_dict['residues'].keys():
            print("Label {} not found in residue list".format(label))
            return

        cts = self.cts[np.array([vec['id'] for vec in self.sim_dict['residues'][label]['ct_vecs']])]
        D = DR.data()
        D.load(Ct={'Ct': cts
            , 't': np.linspace(0, self.length // 1000 * self.universe.trajectory.dt, self.length)})
        n_dets = self.n_dets if self.n_dets else 8
        remove_S2 = 1
        if not hasattr(self,"detect"):
            D.detect.r_auto3(n=n_dets)
            self.detect = D.detect
        else:
            D.detect = self.detect
        fit = D.fit()
        tick_labels = ['~' + tc_str(z0) for z0 in fit.sens.info.loc['z0']]
        tick_labels[0] = '<' + tick_labels[1][1:]
        # tick_labels[-1] = '>' + tick_labels[-2][1:]
        tick_labels = tick_labels[:n_dets - 1]  # removed last label because no data were there
        page_2 = plt.figure()
        page_2.suptitle(label + " - " + self.sel_sim_name)
        h = (cts.shape[0] + 1) // 2 + 1
        page_2.set_size_inches(8.3, 11.7 / 8 * h)  # reduce page size for less cts
        ax0 = page_2.add_subplot(h, 1, 1)
        ax0.plot(D.sens.z(), fit.sens.rhoz().T)
        ax0.set_xlim(-14, -3)
        ax0.set_title("Detector Sensitivities")
        ax0.set_ylabel(r'$(1-S^2)$')
        ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
        ax0.set_yticks([0, 1])
        ax = [page_2.add_subplot(h, 2, 3 + i) for i in range(cts.shape[0])]
        for d in range(n_dets - remove_S2):
            for r in range(cts.shape[0]):
                ax[r].bar(d, fit.R[r, d])
                ax[r].text(d,fit.R[r,d]*1.25,"{:.2f}".format(fit.R[r,d]), ha="center")
                #ax[r].errorbar(d,fit.R[r,d], yerr=fit.R_std[r,d], capsize=5, color="black")
                #plotting errorbars not rally worth
        for r in range(cts.shape[0]):
            maxR = np.max(fit.R[r, :-remove_S2])
            ylim = np.round(maxR + .06, 1) if maxR < .5 else .75 if maxR < .72 else 1
            ax[r].set_ylim(0, ylim)
            ax[r].text(n_dets / 2, ylim * .75, self.sim_dict['residues'][label]['ct_vecs'][r]['name'], bbox=TEXTBOX)
            ax[r].set_xticks(range(n_dets - remove_S2))
            if r < cts.shape[0] - 2:
                ax[r].set_xticklabels([])
            else:
                ax[r].set_xticklabels(tick_labels, rotation=66)
            if r % 2 == 0:
                ax[r].set_ylabel(r'$\rho_n^{(\Theta,S)}$')
        page_2.subplots_adjust(hspace=0.2,right=0.99, left=0.1,bottom=0.15)

    def plot_all(self):
        """this function creates the plots for all examined residues depending on the number of methyl groups and
        if a chi1 and/or chi2 bond is contained. All plots will be saved in a predefined folder (now test/). If only
        a single residue is examined the plot will be immediately shown (but still saved)"""
        def plot_hopability_with_chi_states(axis, index, meth_type, chi_num, chi_type, hoptype="methyl"):
            """there wasnt a very smooth way to put this calculation in the self.calc function, so i put it here
            if a chi2 bond is existing, it will separate the hop probability in three bars to show in which state of
            chi2 the hops are occuring. furthermore, if a shift_axis is given, it will compare if the hop probability
            is changing by the state itself"""
            #todo rename the meth type because now it can be chi or methyl
            #todo this function can be rebuilt anyway
            hop_state_0 = np.zeros(self.n, dtype=self.default_float)
            hop_state_1 = np.zeros(self.n, dtype=self.default_float)
            hop_state_2 = np.zeros(self.n, dtype=self.default_float)
            chi_areas = self.chi1_areas if chi_type == 1 else self.chi2_areas
            hops = self.hops if "methyl" in hoptype else self.hops_chi2 if "chi" in hoptype else None
            for i in range(self.n):
                hop_state_0[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 0)).sum() / part
                hop_state_1[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 1)).sum() / part
                hop_state_2[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 2)).sum() / part
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1 + hop_state_2, color="b")
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1, color="g")
            axis.bar(np.arange(n) + .5, hop_state_0, color="r")
            axis.set_xlim(0, n)
            if hoptype == "methyl":
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                      r'$CH_3^{{{}}} with \chi_{{{}}}-states$'.format(meth_type, chi_type), bbox=TEXTBOX)
            else:
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                          r'$\chi_2 with \chi_1-states$', bbox=TEXTBOX)
            axis.set_xticklabels([])

            #TODO remove
            #if meth_type==2:
            #    axis.set_ylim(0,1)
            #if hoptype != "methyl":
            #    axis.set_ylim(0,0.2)
            #todo remove

        def plot_detectors():
            def tc_str(z):
                unit = 's'
                if z <= -12: z, unit = z + 15, 'fs'
                if z <= -9: z, unit = z + 12, 'ps'
                if z <= -6: z, unit = z + 9, 'ns'
                if z <= -3: z, unit = z + 6, r'$\mu$s'
                if z <= 0: z, unit = z + 3, 'ms'
                tc = np.round(10 ** z, -1 if z >= 2 else (0 if z >= 1 else 1))
                return '{:2} '.format(tc if tc < 10 else int(tc)) + unit
            D = DR.data()
            cts = self.cts[dix[key]["ct_indices"]]
            D.load(Ct={'Ct':cts
                       ,'t': np.linspace(0,int(self.length/1000)*self.universe.trajectory.dt, self.length)})
            n_dets = self.n_dets if self.n_dets else 8
            remove_S2 = 1
            D.detect.r_auto3(n=n_dets)
            fit = D.fit()
            tick_labels = ['~' + tc_str(z0) for z0 in fit.sens.info.loc['z0']]
            tick_labels[0] = '<' + tick_labels[1][1:]
            #tick_labels[-1] = '>' + tick_labels[-2][1:]
            tick_labels = tick_labels[:n_dets-1]  # removed last label because no data were there
            page_2 = plt.figure()
            h = int((cts.shape[0]+1)/2)+1
            page_2.set_size_inches(8.3, 11.7/8*h)  # reduce page size for less cts
            ax0 = page_2.add_subplot(h, 1, 1)
            ax0.plot(D.sens.z(), fit.sens.rhoz().T)
            ax0.set_xlim(-14, -3)
            ax0.set_title("Detector Sensitivities")
            ax0.set_ylabel(r'$(1-S^2)$')
            ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
            ax0.set_yticks([0, 1])
            ax = [page_2.add_subplot(h, 2, 3 + i) for i in range(cts.shape[0])]
            for d in range(n_dets - remove_S2):
                for r in range(cts.shape[0]):
                    ax[r].bar(d, fit.R[r, d])
            for r in range(cts.shape[0]):
                maxR = np.max(fit.R[r, :-remove_S2])
                ylim = np.round(maxR + .06,1) if maxR < .5 else .75 if maxR < .72 else 1
                print(ylim)
                ax[r].set_ylim(0, ylim)
                ax[r].text(n_dets/2, ylim * .75, dix[key]["ct_labels"][r], bbox=TEXTBOX)
                ax[r].set_xticks(range(n_dets - remove_S2))
                if r < cts.shape[0]-2:
                    ax[r].set_xticklabels([])
                else:
                    ax[r].set_xticklabels(tick_labels, rotation=66)
                if r % 2 == 0:
                    ax[r].set_ylabel(r'$\rho_n^{(\Theta,S)}$')
            page_2.suptitle("{}_{}".format(key, self.sel_sim_name))
            page_2.tight_layout()
            pdf.savefig(page_2)
            plt.close(page_2)
        dix = self.full_dict
        part = self.part
        n = self.n
        if not exists(join(self.dir, "plots")):
            mkdir(join(self.dir, "plots"))

        for _, key in enumerate(dix):
            if not "ILE_231" in key:
                continue
            with PdfPages(join(self.dir, "plots", "{}_{}.pdf".format(key, self.sel_sim_name))) as pdf:
                num = dix[key][dix[key]["labels"][0]]["index"]  # index of the first (or only) methylgroup
                if len(dix[key]["labels"]) == 2:
                    two = True
                    num2 = dix[key][dix[key]["labels"][1]]["index"]  # index of the second methylgroup
                else:
                    two = False
                chi1 = dix[key].get("chi1")  # index for chi1
                chi2 = dix[key].get("chi2")  # index for chi2
                ### Initializing figure and axes
                page_1 = plt.figure()
                page_1.set_size_inches(8.3, 11.7)  # (14, 10)
                if chi1 is not None or chi2 is not None:
                    ax_3d_plot = page_1.add_subplot(321, projection='3d')  # for plotting 3d distribution of methyl hydrogens
                    rama_ax = page_1.add_subplot(322)
                    if two:  # check if two methylgroups are available in the residue
                        hopability_methyl_1_axis = page_1.add_subplot(625)  # if yes, create two hopability plots
                        hopability_methyl_2_axis = page_1.add_subplot(627)
                        if chi2 is not None:
                            hopability_methyl_1_axis_B = page_1.add_subplot(6, 2, 9)  # if yes, create two hopability plots
                            hopability_methyl_2_axis_B = page_1.add_subplot(6, 2, 11)
                    else:
                        hopability_single_methyl_axis = page_1.add_subplot(323)  # if not, only one (surprise)
                    if chi2 is not None:  # same here if chi1 and chi2 are available
                        ct_plot = page_1.add_subplot(326)
                        chi1_hopability_plot = page_1.add_subplot(626)  # then make two plots
                        chi2_hopability_plot = page_1.add_subplot(628)
                        pie_chi2 = page_1.add_subplot(645)
                        pie_chi1 = page_1.add_subplot(646)
                        pie_chi2.pie([(self.chi2_areas[chi2]==0).sum(), (self.chi2_areas[chi2]==1).sum(),
                                      (self.chi2_areas[chi2]==2).sum()], colors=["r", "g", "b"])
                        pie_chi2.text(0,0,r'$\chi_2$', ha="center", va="center", bbox=TEXTBOX)
                    else:
                        ct_plot = page_1.add_subplot(313)
                        chi1_hopability_ax_solo = page_1.add_subplot(324)  # or only one
                        #tau_legend.append(r'$\chi_1 hops$')
                        pie_chi1 = page_1.add_subplot(645)
                    pie_chi1.pie([(self.chi1_areas[chi1]==0).sum(), (self.chi1_areas[chi1]==1).sum(),
                                  (self.chi1_areas[chi1]==2).sum()], colors=["r", "g", "b"])
                    pie_chi1.text(0,0,r'$\chi_1$', ha="center", va="center", bbox=TEXTBOX)
                else:
                    # this part is mostly for Alanine
                    ax_3d_plot = page_1.add_subplot(321, projection='3d')
                    ct_plot = page_1.add_subplot(313)
                    hopability_single_methyl_axis = page_1.add_subplot(312)  # if not, only one (surprise)
                ### start to plot, first the 3d scatter plot of the residue chain
                ax_3d_plot.set_axis_off()
                ax_3d_plot.scatter(self.coords_3Dplot[num, :, 0], self.coords_3Dplot[num, :, 1],
                                   self.coords_3Dplot[num, :, 2], s=.5, c="g")  # areas[num])
                carbons = np.array([get_x_y_z_old(dix[key]["C"].position, dix[key]["CA"].position,
                                                  dix[key]["CB"].position, dix[key][_].position) for _ in
                                    dix[key]["chain"]])
                ax_3d_plot.plot(*carbons.T, marker="o", markersize=10)
                for c, label in zip(carbons, dix[key]["chain"]):
                    ax_3d_plot.text(*c.T, label)
                ax_3d_plot.view_init(elev=0, azim=45)
                if len(dix[key]["ct_indices"]):
                    ct_plot.semilogx(np.arange(0, self.length),
                                     self.cts[dix[key]["ct_indices"]].T)
                    for l,index in enumerate(dix[key]["ct_indices"]):
                        dix[key]["ct_labels"][l] += " S²={:.2f}".format(self.S2s[index])
                    ct_plot.legend(dix[key]["ct_labels"])
                    ct_plot.set_xlabel("timepoints")
                    # todo calculate the correlation time by dt and stuff and set the xticks
                    ct_plot.set_ylabel("C(t)")
                    ct_plot.set_yticks([0, .5, 1])
                    ct_plot.set_ylim(-.1, 1)
                    ct_plot.set_title("Correlation functions")

                if two:  # todo this is still clumsy
                    ax_3d_plot.scatter(self.coords_3Dplot[num2, :, 0], self.coords_3Dplot[num2, :, 1],
                                       self.coords_3Dplot[num2, :, 2], s=.5, c="b")  # 3D Rama plot
                    if chi2 is not None:
                        plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi2, 2)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi2, 2)
                        plot_hopability_with_chi_states(hopability_methyl_1_axis_B, num, 1, chi1, 1)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis_B, num2, 2, chi1, 1)
                    elif chi1 is not None:
                        plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi1, 1)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi1, 1)
                    else:
                        hopability_methyl_1_axis.bar(np.arange(n), self.hopability[num, :], color="black")
                        hopability_methyl_2_axis.bar(np.arange(n), self.hopability[num2, :], color="black")
                else:
                    hopability_single_methyl_axis.bar(np.arange(n) + .5, self.hopability[num], color="black")  # hopability
                    hopability_single_methyl_axis.set_xlim(0, n)
                    if chi1 is not None:
                        plot_hopability_with_chi_states(hopability_single_methyl_axis, num, 1, chi1, 1)
                if chi1 is not None or chi2 is not None:
                    if chi2 is not None:
                        rama_ax.hist2d((self.dih_chi1[chi1] + 240) % 360, (self.dih_chi2[chi2] + 240) % 360,
                                       range=[[0, 360], [0, 360]], bins=[180, 180], cmin=0.1)
                        rama_ax.set_xlabel(r'$\chi_1$')
                        rama_ax.set_ylabel(r'$\chi_2$')
                        rama_ax.set_ylim(0, 360)
                        rama_ax.set_yticks([0, 60, 180, 300, 360])
                        chi1_hopability_plot.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                        chi1_hopability_plot.set_xlim(0, n)
                        chi1_hopability_plot.set_xticklabels([])
                        chi1_hopability_plot.text(n / 10, chi1_hopability_plot.get_ylim()[1] * .8,
                                                  r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                        plot_hopability_with_chi_states(chi2_hopability_plot,chi2,2,chi1,1,"chi")
                    else:
                        chi1_hopability_ax_solo.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                        chi1_hopability_ax_solo.set_xlim(0, n)
                        chi1_hopability_ax_solo.text(n / 10, chi1_hopability_ax_solo.get_ylim()[1] * .8,
                                                     r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(0, 120, 3),
                                     range=range(120), color="r")
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(120, 240, 3),
                                     range=range(120), color="b")
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(240, 360, 3),
                                     range=range(120), color="g")
                        rama_ax.set_xlabel(r'$\chi_1$')
                        rama_ax.set_ylabel("occurrence")
                        rama_ax.set_yticklabels([])
                    # final configuration of the plots
                    rama_ax.set_xlim(0, 360)
                    rama_ax.set_xticks([0, 60, 180, 300, 360])

                ct_plot.set_xlim(1, self.length)
                page_1.suptitle("{}_{}".format(key, self.sel_sim_name))
                page_1.tight_layout()
                pdf.savefig(page_1)
                plt.close(page_1)
                plot_detectors()
                print("Save", key, "plot")

    def plot_det_gui(self, residue):
        '''temporary funciton to show the funcitonality of the GUI'''
        def plot_hopability_with_chi_states(axis, index, meth_type, chi_num, chi_type, hoptype="methyl"):
            """there wasnt a very smooth way to put this calculation in the self.calc function, so i put it here
            if a chi2 bond is existing, it will separate the hop probability in three bars to show in which state of
            chi2 the hops are occuring. furthermore, if a shift_axis is given, it will compare if the hop probability
            is changing by the state itself"""
            #todo rename the meth type because now it can be chi or methyl
            #todo this function can be rebuilt anyway
            hop_state_0 = np.zeros(self.n, dtype=self.default_float)
            hop_state_1 = np.zeros(self.n, dtype=self.default_float)
            hop_state_2 = np.zeros(self.n, dtype=self.default_float)
            chi_areas = self.chi1_areas if chi_type == 1 else self.chi2_areas
            hops = self.hops if "methyl" in hoptype else self.hops_chi2 if "chi" in hoptype else None
            for i in range(self.n):
                hop_state_0[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 0)).sum() / part
                hop_state_1[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 1)).sum() / part
                hop_state_2[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 2)).sum() / part
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1 + hop_state_2, color="b")
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1, color="g")
            axis.bar(np.arange(n) + .5, hop_state_0, color="r")
            axis.set_xlim(0, n)
            if hoptype == "methyl":
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                      r'$CH_3^{{{}}} with \chi_{{{}}}-states$'.format(meth_type, chi_type), bbox=TEXTBOX)
            else:
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                          r'$\chi_2 with \chi_1-states$', bbox=TEXTBOX)
            axis.set_xticklabels([])

            #TODO remove
            #if meth_type==2:
            #    axis.set_ylim(0,1)
            #if hoptype != "methyl":
            #    axis.set_ylim(0,0.2)
            #todo remove

        def plot_detectors():
            def tc_str(z):
                unit = 's'
                if z <= -12: z, unit = z + 15, 'fs'
                if z <= -9: z, unit = z + 12, 'ps'
                if z <= -6: z, unit = z + 9, 'ns'
                if z <= -3: z, unit = z + 6, r'$\mu$s'
                if z <= 0: z, unit = z + 3, 'ms'
                tc = np.round(10 ** z, -1 if z >= 2 else (0 if z >= 1 else 1))
                return '{:2} '.format(tc if tc < 10 else int(tc)) + unit
            D = DR.data()
            cts = self.cts[dix[residue]["ct_indices"]]
            D.load(Ct={'Ct':cts
                       ,'t': np.linspace(0,int(self.length/1000)*self.universe.trajectory.dt, self.length)})
            n_dets = self.n_dets if self.n_dets else 8
            remove_S2 = 1
            D.detect.r_auto3(n=n_dets)
            fit = D.fit()
            tick_labels = ['~' + tc_str(z0) for z0 in fit.sens.info.loc['z0']]
            tick_labels[0] = '<' + tick_labels[1][1:]
            #tick_labels[-1] = '>' + tick_labels[-2][1:]
            tick_labels = tick_labels[:n_dets-1]  # removed last label because no data were there
            page_2 = plt.figure()
            h = int((cts.shape[0]+1)/2)+1
            page_2.set_size_inches(8.3, 11.7/8*h)  # reduce page size for less cts
            ax0 = page_2.add_subplot(h, 1, 1)
            ax0.plot(D.sens.z(), fit.sens.rhoz().T)
            ax0.set_xlim(-14, -3)
            ax0.set_title("Detector Sensitivities")
            ax0.set_ylabel(r'$(1-S^2)$')
            ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
            ax0.set_yticks([0, 1])
            ax = [page_2.add_subplot(h, 2, 3 + i) for i in range(cts.shape[0])]
            for d in range(n_dets - remove_S2):
                for r in range(cts.shape[0]):
                    ax[r].bar(d, fit.R[r, d])
            for r in range(cts.shape[0]):
                maxR = np.max(fit.R[r, :-remove_S2])
                ylim = np.round(maxR + .06,1) if maxR < .5 else .75 if maxR < .72 else 1
                print(ylim)
                ax[r].set_ylim(0, ylim)
                ax[r].text(n_dets/2, ylim * .75, dix[residue]["ct_labels"][r], bbox=TEXTBOX)
                ax[r].set_xticks(range(n_dets - remove_S2))
                if r < cts.shape[0]-2:
                    ax[r].set_xticklabels([])
                else:
                    ax[r].set_xticklabels(tick_labels, rotation=66)
                if r % 2 == 0:
                    ax[r].set_ylabel(r'$\rho_n^{(\Theta,S)}$')
            page_2.suptitle("{}_{}".format(residue, self.sel_sim_name))
            page_2.tight_layout()
            return page_2

        dix = self.full_dict
        part = self.part
        n = self.n
        if not exists(join(self.dir, "plots")):
            mkdir(join(self.dir, "plots"))
        num = dix[residue][dix[residue]["labels"][0]]["index"]  # index of the first (or only) methylgroup
        if len(dix[residue]["labels"]) == 2:
            two = True
            num2 = dix[residue][dix[residue]["labels"][1]]["index"]  # index of the second methylgroup
        else:
            two = False
        chi1 = dix[residue].get("chi1")  # index for chi1
        chi2 = dix[residue].get("chi2")  # index for chi2
        ### Initializing figure and axes
        page_1 = plt.figure()
        page_1.set_size_inches(8.3, 11.7)  # (14, 10)
        if chi1 is not None or chi2 is not None:
            ax_3d_plot = page_1.add_subplot(321, projection='3d')  # for plotting 3d distribution of methyl hydrogens
            rama_ax = page_1.add_subplot(322)
            if two:  # check if two methylgroups are available in the residue
                hopability_methyl_1_axis = page_1.add_subplot(625)  # if yes, create two hopability plots
                hopability_methyl_2_axis = page_1.add_subplot(627)
                if chi2 is not None:
                    hopability_methyl_1_axis_B = page_1.add_subplot(6, 2, 9)  # if yes, create two hopability plots
                    hopability_methyl_2_axis_B = page_1.add_subplot(6, 2, 11)
            else:
                hopability_single_methyl_axis = page_1.add_subplot(323)  # if not, only one (surprise)
            if chi2 is not None:  # same here if chi1 and chi2 are available
                ct_plot = page_1.add_subplot(326)
                chi1_hopability_plot = page_1.add_subplot(626)  # then make two plots
                chi2_hopability_plot = page_1.add_subplot(628)
                pie_chi2 = page_1.add_subplot(645)
                pie_chi1 = page_1.add_subplot(646)
                pie_chi2.pie([(self.chi2_areas[chi2]==0).sum(), (self.chi2_areas[chi2]==1).sum(),
                              (self.chi2_areas[chi2]==2).sum()], colors=["r", "g", "b"])
                pie_chi2.text(0,0,r'$\chi_2$', ha="center", va="center", bbox=TEXTBOX)
            else:
                ct_plot = page_1.add_subplot(313)
                chi1_hopability_ax_solo = page_1.add_subplot(324)  # or only one
                #tau_legend.append(r'$\chi_1 hops$')
                pie_chi1 = page_1.add_subplot(645)
            pie_chi1.pie([(self.chi1_areas[chi1]==0).sum(), (self.chi1_areas[chi1]==1).sum(),
                          (self.chi1_areas[chi1]==2).sum()], colors=["r", "g", "b"])
            pie_chi1.text(0,0,r'$\chi_1$', ha="center", va="center", bbox=TEXTBOX)
        else:
            # this part is mostly for Alanine
            ax_3d_plot = page_1.add_subplot(321, projection='3d')
            ct_plot = page_1.add_subplot(313)
            hopability_single_methyl_axis = page_1.add_subplot(312)  # if not, only one (surprise)
        ### start to plot, first the 3d scatter plot of the residue chain
        ax_3d_plot.set_axis_off()
        ax_3d_plot.scatter(self.coords_3Dplot[num, :, 0], self.coords_3Dplot[num, :, 1],
                           self.coords_3Dplot[num, :, 2], s=.5, c="g")  # areas[num])
        carbons = np.array([get_x_y_z_old(dix[residue]["C"].position, dix[residue]["CA"].position,
                                          dix[residue]["CB"].position, dix[residue][_].position) for _ in
                            dix[residue]["chain"]])
        ax_3d_plot.plot(*carbons.T, marker="o", markersize=10)
        for c, label in zip(carbons, dix[residue]["chain"]):
            ax_3d_plot.text(*c.T, label)
        ax_3d_plot.view_init(elev=0, azim=45)
        if len(dix[residue]["ct_indices"]):
            ct_plot.semilogx(np.arange(0, self.length),
                             self.cts[dix[residue]["ct_indices"]].T)
            for l,index in enumerate(dix[residue]["ct_indices"]):
                dix[residue]["ct_labels"][l] += " S²={:.2f}".format(self.S2s[index])
            ct_plot.legend(dix[residue]["ct_labels"])
            ct_plot.set_xlabel("timepoints")
            # todo calculate the correlation time by dt and stuff and set the xticks
            ct_plot.set_ylabel("C(t)")
            ct_plot.set_yticks([0, .5, 1])
            ct_plot.set_ylim(-.1, 1)
            ct_plot.set_title("Correlation functions")

        if two:  # todo this is still clumsy
            ax_3d_plot.scatter(self.coords_3Dplot[num2, :, 0], self.coords_3Dplot[num2, :, 1],
                               self.coords_3Dplot[num2, :, 2], s=.5, c="b")  # 3D Rama plot
            if chi2 is not None:
                plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi2, 2)
                plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi2, 2)
                plot_hopability_with_chi_states(hopability_methyl_1_axis_B, num, 1, chi1, 1)
                plot_hopability_with_chi_states(hopability_methyl_2_axis_B, num2, 2, chi1, 1)
            elif chi1 is not None:
                plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi1, 1)
                plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi1, 1)
            else:
                hopability_methyl_1_axis.bar(np.arange(n), self.hopability[num, :], color="black")
                hopability_methyl_2_axis.bar(np.arange(n), self.hopability[num2, :], color="black")
        else:
            hopability_single_methyl_axis.bar(np.arange(n) + .5, self.hopability[num], color="black")  # hopability
            hopability_single_methyl_axis.set_xlim(0, n)
            if chi1 is not None:
                plot_hopability_with_chi_states(hopability_single_methyl_axis, num, 1, chi1, 1)
        if chi1 is not None or chi2 is not None:
            if chi2 is not None:
                rama_ax.hist2d((self.dih_chi1[chi1] + 240) % 360, (self.dih_chi2[chi2] + 240) % 360,
                               range=[[0, 360], [0, 360]], bins=[180, 180], cmin=0.1)
                rama_ax.set_xlabel(r'$\chi_1$')
                rama_ax.set_ylabel(r'$\chi_2$')
                rama_ax.set_ylim(0, 360)
                rama_ax.set_yticks([0, 60, 180, 300, 360])
                chi1_hopability_plot.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                chi1_hopability_plot.set_xlim(0, n)
                chi1_hopability_plot.set_xticklabels([])
                chi1_hopability_plot.text(n / 10, chi1_hopability_plot.get_ylim()[1] * .8,
                                          r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                plot_hopability_with_chi_states(chi2_hopability_plot,chi2,2,chi1,1,"chi")
            else:
                chi1_hopability_ax_solo.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                chi1_hopability_ax_solo.set_xlim(0, n)
                chi1_hopability_ax_solo.text(n / 10, chi1_hopability_ax_solo.get_ylim()[1] * .8,
                                             r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(0, 120, 3),
                             range=range(120), color="r")
                rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(120, 240, 3),
                             range=range(120), color="b")
                rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(240, 360, 3),
                             range=range(120), color="g")
                rama_ax.set_xlabel(r'$\chi_1$')
                rama_ax.set_ylabel("occurrence")
                rama_ax.set_yticklabels([])
            # final configuration of the plots
            rama_ax.set_xlim(0, 360)
            rama_ax.set_xticks([0, 60, 180, 300, 360])

        ct_plot.set_xlim(1, self.length)
        page_1.suptitle("{}_{}".format(residue, self.sel_sim_name))
        page_1.tight_layout()
        page_2 = plot_detectors()
        return page_1, page_2


def MethylCOrrealation():
    M = KaiMarkov(n=500,  simulation=2, residues=[])
    M.calc(sparse=10)
    M.universe.trajectory[0]
    #M.plot_all()
    #return
    fig = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    bx = fig2.add_subplot(111,projection="3d")
    cx = fig3.add_subplot(111, projection="3d")
    with open("MetRotCorr.bild","w+") as f:
        for key in M.full_dict.keys():
            for label in M.full_dict[key]["labels"]:
                pos = M.full_dict[key][label]["carbon"].position
                ax.scatter(*pos,color="black")
                ax.text(*pos,label)
                num = M.full_dict[key][label]["index"]
                for key2 in M.full_dict.keys():
                    for label2 in M.full_dict[key2]["labels"]:
                        if label ==  label2:
                            continue
                        pos2 = M.full_dict[key2][label2]["carbon"].position
                        num2 = M.full_dict[key2][label2]["index"]
                        corr = np.corrcoef(M.hopability[num],M.hopability[num2])[1][0]
                        if np.linalg.norm(pos-pos2) < 20 and np.abs(corr)>.35:
                            ax.plot([pos[0],pos2[0]],[pos[1],pos2[1]],[pos[2],pos2[2]], color =((np.abs(corr),0,0) if corr < 0 else (0,np.abs(corr),0)),
                                    alpha=corr**2)
                            f.write(".color {} {} {}\n".format(np.abs(corr) if corr < 0 else 0, corr if corr > 0 else 0, 0))
                            #f.write(".transparency {}\n".format(np.abs(corr)))
                            f.write(".cylinder {} {} {} {} {} {} {}\n".format(pos[0],pos[1],pos[2],pos2[0],pos2[1],pos2[2],np.abs(corr-.2)/4))
                chi1 = M.full_dict[key].get("chi1")
                if chi1 is not None:
                    corr = np.corrcoef(M.hopability[num],M.hopability_chi1[chi1])[1][0]
                    bx.scatter(*pos, color=(np.abs(corr),0,0) if corr < 0 else (0,corr,0), s=np.abs(corr)*100)
                    bx.text(*pos, label)
                chi2 = M.full_dict[key].get("chi2")
                if chi2 is not None:
                    corr = np.corrcoef(M.hopability[num],M.hopability_chi2[chi2])[1][0]
                    cx.scatter(*pos, color=(np.abs(corr),0,0) if corr < 0 else (0,corr,0), s=np.abs(corr)*100)
                    cx.text(*pos, label)



def get_detectors_for_simulation():
    M = KaiMarkov(simulation=0)
    #

    indices=[]
    labels = []

    for key in M.full_dict.keys():
        for i, label in enumerate(M.full_dict[key]['ct_labels']):
            if "CH" in label and "rot" in label:
                indices.append(M.full_dict[key]['ct_indices'][i])
                l = ""
                l+= key[4:7]
                l+= key[:3]
                if "1" in label:
                    l+="1"
                elif "2" in label:
                    l+="2"
                else:
                    l+="0"
                labels.append(l)
                print(M.full_dict[key]['ct_indices'][i],label)
    print(indices)
    print(len(indices))
    print(labels)
    indices = np.array(indices)
    M.calc()
    cts = M.cts[indices]
    print(cts.shape)


    D = DR.data()
    D.load(Ct={'Ct': cts
        , 't': np.linspace(0, int(M.length / 1000) * M.universe.trajectory.dt, M.length)})
    n_dets = 8
    D.detect.r_auto3(n=n_dets)
    fit = D.fit()
    fit.label = labels
    print(fit.R.shape)
    marked_atom = {"ala0":"CB",
                   "ile1":"CG2",
                   "ile2":"CD",
                   "leu1":"CD1",
                   "leu2":"CD2",
                   "val1":"CG1",
                   "val2":"CG2",
                   "thr0":"CG2"}

    with open("det2.txt","w+") as f:
        for i,lab in enumerate(fit.label):
            try:
                print(lab)
                f.write(str(int(lab[:3])-147))
                f.write("-")
                f.write(marked_atom[lab[3:].lower()])
                f.write(":")
                for R in fit.R[i,:]:
                    f.write(str(R.astype("float16")))
                    f.write(";")
                f.write("\n")
            except:
                pass


def fractions():
    n = 6
    print(400000//n)
    for i in range(n):
        M = KaiMarkov(simulation=3, offset=400000//n*i,length=400000//n)
        M.calc_new(sparse=0, force_calc=True)
        M.plot_backbone_dynamics()


def plot_ILE231_different_sims():
    detect = None
    for i in range(6):
        M = KaiMarkov(simulation=i, residues=[25],
                      exclude=["met-to-plane", "met-to-plane_2", r'$\chi_{2,lib.}$', r'$\chi_{1,lib.}$'])
        M.calc_new(sparse=0)
        if detect:
            M.detect = detect

        M.plot_single_res("VAL97")
        detect = M.detect

def get_FA():
    mol = DR.molecule(join("pdbs","processed.pdb"), join("xtcs","MET_4pw.xtc"))
    mol.mda_object.residues.resids=mol.mda_object.residues.resids+219-72
    resids = [219]
    mol.select_atoms(sel1="name CG1",sel2="name CD", segids="B",resids=resids)#   Nuc="IVLA",segids="B",resids=resids)
    fr_args = []
    #fr_args.append({'Type': 'hops_3site', 'Nuc': 'ivla', 'segids': 'B','resids':resids, 'sigma': 5, 'ntest': 1000})
    #fr_args.append({'Type': 'methylCC', 'Nuc': 'ivla', 'segids': 'B','resids':resids, 'sigma': 5})
    #fr_args.append({'Type': 'chi_hop', 'Nuc': 'ivla', 'segids': 'B','resids':resids, 'n_bonds': 1, 'sigma': 50, 'ntest': 1000})
    #fr_args.append({'Type': 'side_chain_chi', 'Nuc': 'ivla', 'segids': 'B','resids':resids, 'n_bonds': 1, 'sigma': 50})
    fr_args.append({'Type': 'chi_hop', 'Nuc': 'ivla', 'segids': 'B','resids':resids, 'n_bonds': 2, 'sigma': 50, 'ntest': 1000})
    fr_args.append({'Type': 'side_chain_chi', 'Nuc': 'ivla', 'segids': 'B','resids':resids, 'n_bonds': 2, 'sigma': 50})
    #fr_args.append({'Type': 'peptide_plane', 'segids': 'B','resids':resids, })

    fr_obj=DR.frames.FrameObj(mol)  #This creates a frame object based on the above molecule object
    fr_obj.tensor_frame(sel1=1,sel2=2) #Here we specify to use the same bonds that were selected above in mol.select_atoms

    for f in fr_args:fr_obj.new_frame(**f) #This defines all frames in the frame object

    fr_obj.load_frames(n=-1, tf=10000)  #This sweeps through the trajectory and collects the directions of all frames at each time point
    t0=time()
    fr_obj.post_process()   #This applies post processing to the frames where it is defined (i.e. sigma!=0)
    print(time()-t0) #261 seconds
    t0=time()
    data=fr_obj.frames2data()
    for d in data:
        plt.figure()
        plt.semilogx(d.R.T)
    return data

@time_runtime
def main():
    #get_FA()

    M= KaiMarkov(simulation=3)#, residues=np.array([72,75,84,94,103,107,109,129,130])-72)

    M.calc(sparse=0)#, force_calculation=True)
    #M.plot_single_res("ILE72", 1)
    for i in range(9):
        M.do_markov(i)
        M.do_markov_3dihedrals(i)
    exit()
    for i in range(9):
        M.do_markov(i)
    exit()
    #M.do_markov_3dihedrals(0)
    #plt.show()
    #plt.show()
    #plot_ILE231_different_sims()
    return
    for i in range(1):
        M = KaiMarkov(simulation=i)
        M.calc_new(sparse=0)
        print(M.sim_dict["residues"])
        M.plot_single_res("ILE72")
    plt.show()

    #for key in M.sim_dict['residues'].keys():
    #    M.plot_single_res(key)


if __name__ == "__main__":
    main()
