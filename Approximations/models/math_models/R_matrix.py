import numpy as np
from copy import deepcopy

from Approximations.rmatrix.base.particles import Particle
from Approximations.rmatrix.channels.elastic_channel import ElasticChannel
from Approximations.rmatrix.channels.capture_channel import CaptureChannel
from Approximations.rmatrix.spin_group import SpinGroup

class RM_Interface():
    def __init__(self,num_channels,num_levels) -> None:
        self.num_channels=num_channels
        self.num_levels=num_levels
        self.incoming_particle=None
        self.outgoing_particle=None
        self.target_particle=None
        self.compound_particle=None
        self.elastic_channel=None
        self.capture_channels=[]
        self.res_energies=[]
    def set_incoming(self,incoming_particle:Particle)->None:
        self.incoming_particle=incoming_particle
    def set_outgoing(self,outgoing_particle:Particle)->None:
        self.outgoing_particle=outgoing_particle
    def set_target(self,target_particle:Particle)->None:
        self.target_particle=target_particle
    def set_compound(self,compound_particle:Particle)->None:
        self.compound_particle=compound_particle
    
    def set_elastic_channel(self,J:int,pi:int,ell:int,radius:float,reduced_width_amplitudes:list)->None:
        assert not(self.incoming_particle==None), "No incoming particle defined in elastic channel."
        assert not(self.target_particle==None), "No target particle defined in elastic channel."
        assert not(len(reduced_width_amplitudes)<self.num_levels), "Not enough resonances in elastic channel."
        assert not(len(reduced_width_amplitudes)>self.num_levels), "Too many resonances in elastic channel."
        self.elastic_channel=ElasticChannel(self.incoming_particle,
                                            self.target_particle,
                                            J,
                                            pi,
                                            ell,
                                            radius,
                                            reduced_width_amplitudes)
    def get_elastic_channel(self)->ElasticChannel:
        return(self.elastic_channel)
    
    def add_capture_channel(self,J:int,pi:int,ell:int,radius:float,reduced_width_amplitudes:list,excitation:float)->None:
        assert not(self.outgoing_particle==None), "No outgoing particle defined in capture channel."
        assert not(self.compound_particle==None), "No compound particle defined in capture channel."
        assert not(len(reduced_width_amplitudes)<self.num_levels), "Not enough resonances in capture channel."
        assert not(len(reduced_width_amplitudes)>self.num_levels), "Too many resonances in capture channel."
        self.capture_channels.append(CaptureChannel(self.outgoing_particle,
                                                    self.compound_particle,
                                                    J,
                                                    pi,
                                                    ell,
                                                    radius,
                                                    reduced_width_amplitudes,
                                                    excitation))
    def get_capture_channels(self)->list:
        return(self.capture_channels)
    def remove_capture_channel(self,index:int)->None:
        del self.capture_channels[index]
    def clear_capture_channels(self)->None:
        self.capture_channels=[]
    
    def set_energy_grid(self,energy_grid:np.array)->None:
        self.energy_grid=energy_grid
    def get_energy_grid(self)->np.array:
        return(self.energy_grid)
    
    def set_resonance_energies(self,energies:list)->None:
        assert not(len(energies)<self.num_levels), "Not enough resonance energies."
        assert not(len(energies)>self.num_levels), "Too many resonance energies."
        self.res_energies=energies
    def get_resonance_energies(self)->list:
        return(self.res_energies)
    
    def establish_spin_group(self)->None:
        assert len(self.res_energies)>0, "Resonance energies not set."
        assert not(self.elastic_channel==None), "Elastic channel not set."
        assert not(len(self.capture_channels)<(self.num_channels-1)), "Not enough capture channels set."
        assert not(len(self.capture_channels)>(self.num_channels-1)), "too many capture channels set."
        assert len(self.energy_grid)>0, "Energy grid not set."
        self.spin_group=SpinGroup(self.res_energies, self.elastic_channel, self.capture_channels,self.energy_grid)
        self.spin_group.calc_cross_section()
    def get_spin_group(self)->SpinGroup:
        return(self.spin_group)
    
    def set_gamma_matrix(self,gamma_matrix:np.array)->None:
        assert not(gamma_matrix.shape==[self.num_levels,self.num_channels]),"Gamma matrix is the wrong shape."
        self.spin_group.update_gamma_matrix(gamma_matrix)
    def get_gamma_matrix(self)->np.array:
        return(np.copy(self.spin_group.gamma_matrix))
    
    def get_L_matrix(self)->np.array:
        return(self.spin_group.L_matrix)
    def get_cross_section(self)->np.array:
        return(self.spin_group.total_cross_section)
    def get_channels(self)->list:
        return(self.spin_group.channels)
    
    def get_data(self, gamma_matrix=None)->np.array:
        if(gamma_matrix is None):
            test_spin_group=self.spin_group
        else:
            test_spin_group = deepcopy(self.spin_group)
            test_spin_group.update_gamma_matrix(gamma_matrix)

        data    = [np.zeros(len(self.energy_grid)), np.zeros((self.num_channels, len(self.energy_grid)))]
        data[0] = test_spin_group.total_cross_section
        for idx in range(self.num_channels):
            data[1][idx,:] = test_spin_group.channels[idx].cross_section

        return(data)