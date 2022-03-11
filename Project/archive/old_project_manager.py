#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:27:10 2022

@author: albertsmith
"""

class Project1():
    def __init__(self, directory, create=False, subproject=False):
        self.name = directory   #todo maybe create the name otherwise?
        self._directory = os.path.abspath(directory)
        if not(os.path.exists(self.directory)) and create:
            os.mkdir(self.directory)
        assert os.path.exists(self.directory),'Project directory does not exist. Select an existing directory or set create=True'
               
        self.data=DataMngr(self)
        self.__subproject = subproject  #Subprojects cannot be edited/saved
        self.plots = [None]
        self._current_plot = [0]
    
    
    @property
    def fig(self):
        if self.current_plot:
            return self.plots[self.current_plot-1].fig
        
    def plot(self, data_index=None, data=None, fig=None, style='plot',
                  errorbars=False, index=None, rho_index=None, split=True, plot_sens=True, **kwargs):
        """
        

        Parameters
        ----------
        data : pyDR.Data, optional
            data object to be plotted. The default is None.
        data_index : int, optional
            index to determine which data object in the project to plot. The default is None.
        fig : int, optional
            index to determine which plot to use. The default is None (goes to current plot).
        style : str, optional
            'p', 's', or 'b' specifies a line plot, scatter plot, or bar plot. The default is 'plot'.
        errorbars : bool, optional
            Show error bars (True/False). The default is True.
        index : int/bool array, optional
            Index to determine which residues to plot. The default is None (all residues).
        rho_index : int/bool array, optional
            index to determine which detectors to plot. The default is None (all detectors).
        split : bool, optional
            Split line plots with discontinuous x-data. The default is True.
        plot_sens : bool, optional
            Show the sensitivity of the detectors in the top plot (True/False). The default is True.
        **kwargs : TYPE
            Various arguments that are passed to matplotlib.pyplot for plotting (color, linestyle, etc.).

        Returns
        -------
        None.

        """
        if data is None and data_index is None: #Plot everything in project
            for i in range(self.size):
                self.plot(data_index=i,style=style,errorbars=errorbars,index=index,
                         rho_index=rho_index,plot_sens=plot_sens,split=split,fig=fig,**kwargs)
            return
        
        if fig is None and self.current_plot==0:self.current_plot=1
        
        fig = self.current_plot-1 if fig is None else fig-1
        self._current_plot[0] = fig+1
        if fig is not None:
            while len(self.plots) <= fig:
                self.plots.append(None)
        if data is None:
            data = self[data_index]
        if self.plots[fig] is None:
            self.plots[fig] = clsDict['DataPlots'](data=data, style=style, errorbars=errorbars, index=index,
                         rho_index=rho_index, plot_sens=plot_sens, split=split, **kwargs)
            self.plots[fig].project=self
        else:
            self.plots[fig].append_data(data=data,style=style,errorbars=errorbars,index=index,
                         rho_index=rho_index,plot_sens=plot_sens,split=split,**kwargs)
    @property
    def current_plot(self):
        return self._current_plot[0]
    
    def close_fig(self, fig):
        """
        Closes a figure of the project

        Parameters
        ----------
        plt_num : int
            Clears and closes a figure in the project. Provide the index 
            (ignores indices corresponding to non-existant figures)

        Returns
        -------
        None.

        """
        if isinstance(fig,str) and fig.lower()=='all':
            for i in range(len(self.plots)):self.close_fig(i)
            return
        fig-=1
        if len(self.plots) > fig and self.plots[fig] is not None:
            self.plots[fig].close()
            self.plots[fig] = None
            

    @property
    def subproject(self):
        return self.__subproject
    
    @property
    def directory(self):
        return self._directory
    
    def append_data(self,data):
        assert not(self.__subproject),"Data cannot be appended to subprojects"
        self.data.append_data(data)
    def remove_data(self,index,delete=False):
        assert not(self.__subproject),"Data cannot be removed from subprojects"
        proj=self[index]
        if hasattr(proj,'R'):
            self.data.remove_data(index=self.data.data_objs.index(proj),delete=delete)
        else:
            if delete:print('Delete data sets permanently by full title or index (no multi-delete of saved data)')
            for d in proj:        
                self.data.remove_data(index=self.data.data_objs.index(d))
        
    def save(self):
        assert not(self.__subproject),"Sub-projects cannot be saved"
        self.data.save()
    
    @property
    def detectors(self):
        # TODO I was wondering if there might be a better way to achieve this, generally it seems that
        #  r = list(dict.fromkeys(r0))
        #  should do it
        #  return list(dict.fromkeys(r0))
        #  might be the shortest, but I am actually not sure if that works with objects?    -K
        r=list()
        for r0 in self.data.detect_list:
            if r0 not in r:
                r.append(r0)
        return r
        
    def unify_detect(self, chk_sens_only: bool = False) ->None:
        """
        Checks for equality among the detector objects and assigns all equal
        detectors to the same object. This allows one to only optimize one of
        the data object's detectors, and all other objects with equal detectors
        will automatically be optimized.
        
        Note that by default, two detectors with differently defined sensitivities
        will not be considered equal. However, one may set chk_sens_only=True in
        which case only the sensitivity of the detectors will be checked, so
        all detectors having the same initial sensitivity will be combined.
        """
        r = self.data.detect_list
        s = [r0.sens for r0 in r]
        for k, (s0, r0) in enumerate(zip(s, r)):
            if (s0 in s[:k]) if chk_sens_only else (r0 in r[:k]):                
                i=s[:k].index(s0) if chk_sens_only else r[:k].index(r0)
                self.data[k].sens = s[i]
                self.data[k].detect = r[i]
                
    def unique_detect(self, index: int = None) -> None:
        if index is None:
            for i in range(len(self.data)):
                self.unique_detect(i)
        else:
            d = self.data[index]
            sens = d.detect.sens
            d.detect = d.detect.copy()
            d.detect.sens = sens
    
    def __iter__(self):
        def gen():
            for k in range(len(self.data)):
                yield self.data[k]
        return gen()
    
    def __len__(self) -> int:
        return self.data.__len__()
    
    def __setattr__(self,name,value):
        if name=='current_plot':
            self._current_plot[0]=value
            while len(self.plots)<value:
                self.plots.append(None)
            self.plots[value-1]=clsDict['DataPlots']()
            return
        super().__setattr__(name,value)

    def _ipython_display_(self):
        print("pyDIFRATE project with {0} data sets\n{1}\n".format(self.size,super().__repr__()))
        print('Titles:')
        for t in self.titles:print(t)
    @property
    def size(self) -> int:
        return self.__len__()
    
    def __getitem__(self, index: int):
        """
        Extract a data object or objects by index or title (returns one item) or
        by Type or status (returns a list).
        """
        if isinstance(index, int): #Just return the data object
            assert index < self.__len__(), "index too large for project of length {}".format(self.__len__())
            return self.data[index]
        
        #Otherwise, return a subproject
        proj=Project(self.directory, create=False, subproject=True)
        proj.plots = self.plots
        proj._current_plot = self._current_plot

        data = list()
        if isinstance(index, str):
            if index in self.Types:
                for k, t in enumerate(self.Types):
                    if index == t:
                        data.append(self[k])
            elif index in self.statuses:
                for k,s in enumerate(self.statuses):
                    if index == s:
                        data.append(self[k])
            elif index in self.add_info:
                for k, s in enumerate(self.add_info):
                    if index == s:
                        data.append(self[k])
            elif index in self.titles:
                return self[self.titles.index(index)]
            elif index in self.short_files:
                for k, s in enumerate(self.short_files):
                    if index == s:
                        data.append(self[k])
            else:
                r = re.compile(index)
                for k,t in enumerate(self.titles):
                    if r.match(t):data.append(self[k])
                if not(len(data)):
                    print('Unknown project index')
                    return
        elif hasattr(index, '__len__'):
            for k, s in enumerate(self.add_info):
                if index == s:
                    data.append(self[k])     # todo I am not sure what you try to achieve here
            data = [self[i] for i in index]  #  but it looks wrong   -K
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step
            stop = self.size if index.stop is None else min(index.stop, self.size)
            start %= self.size
            stop = (stop-1) % self.size+1
            if step<0:start,stop=stop-1,start-1
            data = [self[i] for i in range(start, stop, step)]
        else:
            print('index was not understood')
            return
    
        proj.data = DataSub(data[0].source.project, *data)
        return proj
    
    def _ipython_key_completions_(self) -> list:
        out = list()
        for k in ['Types', 'statuses', 'add_info', 'titles']:
            for v in getattr(self, k):
                if v not in out:
                    out.append(v)
        return out

    def opt2dist(self, rhoz_cleanup=False, parallel=False) -> None:
        """
        Optimize fits to match a distribution for all detectors in the project.
        """
        sens = list()
        detect = list()
        count = 0
        for d in self:
            if hasattr(d.sens,'opt_pars') and 'n' in d.sens.opt_pars:
                fit = d.opt2dist(rhoz_cleanup, parallel=parallel)
                if fit is not None:
                    count += 1
                    if fit.sens in sens:
                        i = sens.index(fit.sens)
                        fit.sens = sens[i]
                        fit.detect = detect[i]
                    else:
                        sens.append(fit.sens)
                        detect.append(clsDict['Detector'](fit.sens))
        print('Optimized {0} data objects'.format(count))
    
    def fit(self, bounds: bool = True, parallel: bool = False) -> None:
        """
        Fit all data in the project that has optimized detectors.
        """
        sens = list()
        detect = list()
        count = 0
        for d in self:
            if 'n' in d.detect.opt_pars:
                count += 1
                fit = d.fit(bounds=bounds, parallel=parallel)
                if fit.sens in sens:
                    i = sens.index(fit.sens)    #We're making sure not to have copies of identical sensitivities and detectors
                    fit.sens = sens[i]
                    fit.detect = detect[i]
                else:
                    sens.append(fit.sens)
                    detect.append(fit.detect)
        print('Fitted {0} data objects'.format(count))
    
    @property
    def Types(self):
        return [d.source.Type for d in self]
    
    @property
    def statuses(self):
        return [d.source.status for d in self]
    
    @property
    def titles(self): 
        return [d.title for d in self]
    
    @property
    def short_files(self):
        return [d.source.short_file for d in self]
    
    @property
    def add_info(self):
        return [d.source.additional_info for d in self]
    
    def comparable(self, i: int, threshold: float = 0.9, mode: str = 'auto', min_match: int = 2) -> tuple:
        """
        Find objects that are recommended for comparison to a given object. 
        Provide either an index (i) referencing the data object's position in 
        the project or by providing the object itself. An index will be
        returned, corresponding to the data objects that should be compared to
        the given object
        
        Comparability is defined as:
            1) Having some selections in common. This is determined by running 
                self[k].source.select.compare(i,mode=mode)
            2) Having 
                a) Equal sensitivities (faster calculation, performed first)
                b) overlap of the object sensitivities above some threshold.
                This is defined by sens.overlap_index, where we require 
                at least min_match elements in the overlap_index (default=2)
        """

        #todo argument mode is never used?

        if isinstance(i, int):
            i = self[i] #Get the corresponding data object
        out = list()
        for s in self:
            if s.select.compare(i.select)[0].__len__() == 0:
                out.append(False)
                continue
            out.append(s.sens.overlap_index(i.sens, threshold=threshold)[0].__len__() >= min_match)
        return np.argwhere(out)[:, 0]