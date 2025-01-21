import numpy as np
from copy import deepcopy

from Approximations.rmatrix.base.particles import Particle
from Approximations.rmatrix.channels.elastic_channel import ElasticChannel
from Approximations.rmatrix.channels.capture_channel import CaptureChannel
from Approximations.rmatrix.spin_group import SpinGroup

class Fundamental():
    def __init__(self,num_channels,num_resonances) -> None:
        self.num_channels=num_channels
        self.num_resonances=num_resonances
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
        assert not(len(reduced_width_amplitudes)<self.num_resonances), "Not enough resonances in elastic channel."
        assert not(len(reduced_width_amplitudes)>self.num_resonances), "Too many resonances in elastic channel."
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
        assert not(len(reduced_width_amplitudes)<self.num_resonances), "Not enough resonances in capture channel."
        assert not(len(reduced_width_amplitudes)>self.num_resonances), "Too many resonances in capture channel."
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
        assert not(len(energies)<self.num_resonances), "Not enough resonance energies."
        assert not(len(energies)>self.num_resonances), "Too many resonance energies."
        self.res_energies=energies
    def get_resonance_energies(self)->list:
        return(self.res_energies)
    
    def establish_spin_group(self)->None:
        assert len(self.res_energies)>0, "Resonance energies not set."
        assert not(self.elastic_channel==None), "Elastic channel not set."
        assert not(len(self.capture_channels)<self.num_resonances), "Not enough capture channels set."
        assert not(len(self.capture_channels)>self.num_resonances), "too many capture channels set."
        assert len(self.energy_grid)>0, "Energy grid not set."
        self.spin_group=SpinGroup(self.res_energies, self.elastic_channel, self.capture_channels,self.energy_grid)
        self.spin_group.calc_cross_section()
    def get_spin_group(self)->SpinGroup:
        return(self.spin_group)
    
    def set_gamma_matrix(self,gamma_matrix:np.array)->None:
        assert not(gamma_matrix.shape==[self.num_resonances,self.num_channels]),"Gamma matrix is the wrong shape."
        self.spin_group.update_gamma_matrix(gamma_matrix)
    def get_gamma_matrix(self)->np.array:
        return(np.copy(self.spin_group.gamma_matrix))
    
    def get_L_matrix(self)->np.array:
        return(self.spin_group.L_matrix)
    def get_cross_section(self)->np.array:
        return(self.spin_group.total_cross_section)
    def get_channels(self)->list:
        return(self.spin_group.channels)
    
    
    
    def derivative_of_U_matrix(self,spin_group,gamma_der):
        A_der=-(gamma_der@spin_group.L_matrix@spin_group.gamma_matrix.T+spin_group.gamma_matrix@spin_group.L_matrix@gamma_der.T)
        A_inv_der=-(spin_group.A_matrix@A_der@spin_group.A_matrix)
        W_der=2j*spin_group.P_half@(gamma_der.T@spin_group.A_matrix@spin_group.gamma_matrix+
                     spin_group.gamma_matrix.T@A_inv_der@spin_group.gamma_matrix+
                     spin_group.gamma_matrix.T@spin_group.A_matrix@gamma_der)@spin_group.P_half
        U_der=spin_group.omega_matrix@W_der@spin_group.omega_matrix
        return(U_der)
    
    def derivative_of_elastic_channel(self,spin_group,U_der):
        chan_der=10**24 * np.pi/spin_group.k_sq*(-2*U_der[:,0,0].real+np.conjugate(U_der[:,0,0])*spin_group.U_matrix[:,0,0]+np.conjugate(spin_group.U_matrix[:,0,0])*U_der[:,0,0]).real
        return(chan_der)
    
    def derivative_of_capture_channel(self,spin_group,U_der,channel_num):
        chan_der=10**24 * np.pi/spin_group.k_sq *(np.conjugate(U_der[:,0,channel_num])*spin_group.U_matrix[:,0,channel_num]+np.conjugate(spin_group.U_matrix[:,0,channel_num])*U_der[:,0,channel_num]).real
        return(chan_der)