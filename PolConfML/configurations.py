import numpy as np
from .utils import _periodic_distance, _rescale_and_pad

class Configuration_TiO2():
    def __init__(self, polaron_positions, defect_positions, box_size):
        self.positions = polaron_positions
        self.defects = defect_positions
        self.box_size = box_size
        self.layer = self.layer_index()
        self.site = self.site_index()
        self.site['A'] = np.concatenate(self.site['A1']+self.site['A2'])
        self.site['B'] = np.concatenate(self.site['B1']+self.site['B2'])
    
    def layer_index(self):
        idx = {}
        idx['S0'] = (self.positions[:,2]>12).nonzero()
        idx['S1'] = np.logical_and(self.positions[:,2]<12,self.positions[:,2]>10).nonzero()
        idx['S2'] = (self.positions[:,2]<10).nonzero()
        self.layer = idx
        return idx
    
    def site_index(self):
        y = 13.1824998856
        y_8 = y/8
        upper = [1*y_8,3*y_8,5*y_8,7*y_8]
        lower = [7*y_8,1*y_8,3*y_8,5*y_8]
        idx = {}
        
        idx['A1'] = np.logical_or(self.positions[:,1]>lower[0],
                                  self.positions[:,1]<upper[0]).nonzero()
        idx['B1'] = np.logical_and(self.positions[:,1]>lower[1],
                                  self.positions[:,1]<upper[1]).nonzero()
        idx['A2'] = np.logical_and(self.positions[:,1]>lower[2],
                                  self.positions[:,1]<upper[2]).nonzero()
        idx['B2'] = np.logical_and(self.positions[:,1]>lower[3],
                                  self.positions[:,1]<upper[3]).nonzero()
        return idx
    
    def __getitem__(self, location):
        layer,site = location
        common_idx = np.intersect1d(self.layer[layer],self.site[site])
        return self.positions[common_idx]
    
    def calc_desc(self, from_layer, from_site, n, R_c):
        layers = ['S0','S1','S2']
        polarons = self[from_layer,from_site]
        descriptors = []
        for polaron in polarons:
            descriptor = []
            for layer in layers:
                if from_site == 'A1' or from_site =='A2':
                    # distance to stacked same site polarons
                    distances = _periodic_distance(polaron, 
                                               self[layer,from_site],
                                               self.box_size)
                    distances = _rescale_and_pad(distances,n)
                    descriptor.append(distances)
                    
                    #distance to non stacked same site polarons
                    if from_site == 'A1':
                        distances = _periodic_distance(polaron, 
                                                   self[layer,'A2'],
                                                   self.box_size)
                        distances = _rescale_and_pad(distances,n)
                        descriptor.append(distances)
                        
                    if from_site == 'A2':
                        distances = _periodic_distance(polaron, 
                                                   self[layer,'A1'],
                                                   self.box_size)
                        distances = _rescale_and_pad(distances,n)
                        descriptor.append(distances)
                    
                    # distance to different site coordination
                    distances = _periodic_distance(polaron, 
                                                   self[layer,'B'],
                                                   self.box_size)
                    distances = _rescale_and_pad(distances,n)
                    descriptor.append(distances)
                    
                    
                if from_site == 'B1' or from_site =='B2':
                    # distance to stacked same site polarons
                    distances = _periodic_distance(polaron, 
                                                   self[layer,from_site],
                                                   self.box_size)
                    distances = _rescale_and_pad(distances,n)
                    descriptor.append(distances)
                    
                    # distance to non stacked same site polarons
                    if from_site == 'B1':
                        distances = _periodic_distance(polaron, 
                                                   self[layer,'B2'],
                                                   self.box_size)
                        distances = _rescale_and_pad(distances,n)
                        descriptor.append(distances)
                    if from_site == 'B2':
                        distances = _periodic_distance(polaron, 
                                                   self[layer,'B1'],
                                                   self.box_size)
                        distances = _rescale_and_pad(distances,n)
                        descriptor.append(distances)
                    
                    # distance to different site coordination
                    distances = _periodic_distance(polaron, 
                                                   self[layer,'A'],
                                                   self.box_size)
                    distances = _rescale_and_pad(distances,n)
                    descriptor.append(distances)
                    
            # defect distance
            distances = _periodic_distance(polaron,
                                          self.defects,
                                          self.box_size)
            distances = _rescale_and_pad(distances,n, R_c)
            descriptor.append(distances)
            descriptors.append(np.concatenate(descriptor))
        if not descriptors:
            return np.array([]).reshape(0,10*n)
        return np.array(descriptors)
    
    def full_descriptors(self,n,idx,R_c):
        sites = ['A1','B1','A2','B2']
        descriptors = {
            'S0_A': np.concatenate(
                [self.calc_desc('S0','A1',n,R_c),
                 self.calc_desc('S0','A2',n,R_c)]
            ),
            'S1_A':np.concatenate(
                [self.calc_desc('S1','A1',n,R_c),
                 self.calc_desc('S1','A2',n,R_c)]
            ),
            'S0_B':np.concatenate(
                [self.calc_desc('S0','B1',n,R_c),
                 self.calc_desc('S0','B2',n,R_c)]
            ),
            'S1_B':np.concatenate(
                [self.calc_desc('S1','B1',n,R_c),
                 self.calc_desc('S1','B2',n,R_c)]
            ),
            'S2_A':np.concatenate(
                [self.calc_desc('S2','A1',n,R_c),
                 self.calc_desc('S2','A2',n,R_c)]
            ),
            'S2_B':np.concatenate(
                [self.calc_desc('S2','B1',n,R_c),
                 self.calc_desc('S2','B2',n,R_c)]
            )
        }
        
        idxs = {
            'S0_A': np.ones(descriptors['S0_A'].shape[0])*idx,
            'S0_B': np.ones(descriptors['S0_B'].shape[0])*idx,
            'S1_A': np.ones(descriptors['S1_A'].shape[0])*idx,
            'S1_B': np.ones(descriptors['S1_B'].shape[0])*idx,
            'S2_A': np.ones(descriptors['S2_A'].shape[0])*idx,
            'S2_B': np.ones(descriptors['S2_B'].shape[0])*idx,
        }
        return descriptors, idxs

class Configuration_SrTiO3():
    """ Class that classifies polarons for the specific super cell from the MD-database
    allows calculation of descriptor by calling method full_descriptors(...)

    Attributes
    ----------
    polaron_positions : numpy.array - cartesian coordinates of the polarons
    defect_positions : numpy.array - cartesian coordinates of the defects

    Methods
    -------
    layer_index() : assigns layer index to each polaron
    site_index() : assigns site index to each polaron
    __getitem__() : returns polarons at specified site and layer
    calc_desc(from_layer, from_site, n, R_c) : calculates descriptors for polarons 
                                               in specified layer and site
    full_descriptors(n, idx, R_c) : wrapper that automates descriptor calculation for 
                                    entire configuration
    """
    def __init__(self, polaron_positions, defect_positions, box_size):
        self.positions = polaron_positions
        self.defect_positions = defect_positions
        self.box_size = box_size
        self.idx_pol, self.idx_def = self.layer_index()
    
    def layer_index(self):
        idx_pol = {}
        idx_pol['S0'] = (self.positions[:,2]>self.box_size[2]*0.5).nonzero()
        idx_pol['S1'] = np.logical_and(self.positions[:,2]<self.box_size[2]*0.5,
                                       self.positions[:,2]>self.box_size[2]*0.4).nonzero()
        idx_pol['S2'] = np.logical_and(self.positions[:,2]<self.box_size[2]*0.4,
                                       self.positions[:,2]>self.box_size[2]*0.3).nonzero()
        idx_def = {}
        idx_def['S0'] = (self.defect_positions[:,2]>self.box_size[2]*0.5).nonzero()
        idx_def['S1'] = np.logical_and(self.defect_positions[:,2]<self.box_size[2]*0.5,
                                       self.defect_positions[:,2]>self.box_size[2]*0.4).nonzero()
        idx_def['S2'] = np.logical_and(self.defect_positions[:,2]<self.box_size[2]*0.4,
                                       self.defect_positions[:,2]>self.box_size[2]*0.3).nonzero()
        return idx_pol, idx_def
 
    
    def __getitem__(self, arg):
        pol, layer = arg
        if pol:
            return self.positions[self.idx_pol[layer]]
        else:
            return self.defect_positions[self.idx_def[layer]]
    
    def calc_desc(self, from_layer, n, R_c=15):
        """ calculates descriptors specific to the polaron hosting site 

        Depending on from_layer and from site a differently structured descriptor
        array is returned

        Parameters
        ----------
        from_layer : str
            specifies layer of polarons that should be considered from 
            ['S0', 'S1', 'S2']
        from_site : str
            specifies site of polarons that should be considered from
            ['A1', 'A2', 'B1','B2']
        n : int
            number of included descriptor in specific interaction category
        R_c : float 
            cut off radius

        Returns
        -------
        descriptors : numpy.array
            array containing all descriptor vectors for polarons at specified hosting site
            in the given configuration
        """
        layers = ['S0','S1','S2']
        polarons = self[True,from_layer]
        descriptors = []
        for polaron in polarons:
            descriptor = []
            for layer in layers:
                distances = _periodic_distance(polaron, 
                                           self[True,layer],
                                           self.box_size)
                distances = _rescale_and_pad(distances, n, R_c)
                descriptor.append(distances)
                
                    
                distances = _periodic_distance(polaron, 
                                           self[False,layer],
                                           self.box_size)
                distances = _rescale_and_pad(distances, n, R_c)
                descriptor.append(distances)
            descriptors.append(np.concatenate(descriptor))
        if not descriptors:
            return np.array([]).reshape(0, 6*n)
        return np.array(descriptors)
    
    def full_descriptors(self, n, idx, R_c=15):
        """ calculates full descriptors for all polarons in configuration
        
        Parameters
        ----------
        n : int - number of distances in each interaction category
        idx : int - index of configuration that allows later linkage of a polaron
                    descriptor to a specific configuration
        R_c : float - cutoff radius

        Returns
        -------
        descriptors : dict - dictionary containing all descriptors at site; keys are site 
                             and layer combinations
        idxs : dict - dictionary containing indices for all single polaron descriptors
        """
        descriptors = {
            'S0': np.concatenate(
                [self.calc_desc('S0',n, R_c)]
            ),
            'S1': np.concatenate(
                [self.calc_desc('S1',n, R_c)]
            ),
            'S2': np.concatenate(
                [self.calc_desc('S2',n, R_c)]
            )
        }
        idxs = {
            'S0': np.ones(descriptors['S0'].shape[0])*idx,
            'S1': np.ones(descriptors['S1'].shape[0])*idx,
            'S2': np.ones(descriptors['S2'].shape[0])*idx,
        }
        return descriptors, idxs