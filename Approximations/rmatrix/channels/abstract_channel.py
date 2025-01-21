from abc import ABC, abstractmethod
import numpy as np

class AbstractChannel(ABC):

    def __init__(self,light_product,heavy_product,J,pi,ell,ac,reduced_width_aplitudes, excitation=0):
        """ Abstract class representing a single channel
        
        Parameters
        ----------
        light_product : Particle object
            The light product: the neutron for elastic, 
            or the gamma for capture

        heavy_product : Particle object
            The heavy product in the channel

        J : float
            The spin of the channel

        pi : int
            The parity of the channel, 1 or -1

        ell : int
            Orbital angular momentum for the channel

        ac : float
            The channel radius in 10^(-12) cm

        reduced_width_aplitudes : list or numpy array
            Reduced width amplitues for the resonances in the 
            spin group
        
        excitation  : float, optional, default is 0
            The excitation  energy of the heavy nucleus after
            the reaction, in eV


        Attributes
        ----------
        light_product : Particle object
            The light product: the neutron for elastic, 
            or the gamma for capture

        heavy_product : Particle object
            The heavy product in the channel

        A : int
            Mass number of the compound nucleus

        J : float
            The spin of the channel

        pi : int
            The parity of the channel, 1 or -1

        ell : int
            Orbital angular momentum for the channel

        ac : float
            The channel radius in 10^(-12) cm

        excitation  : float
            The excitation energy of the heavy nucleus after
            the reaction, in eV

        reduced_width_aplitudes : numpy array
            Reduced width amplitues for the resonances in the 
            spin group


        Methods
        -------

        calc_k
            Function to calculate k for the channel 
        
        calc_rho
            Function to calculate rho for the channel 
        
     
        calc_penetrability
            Function to calculate the penetrability 
            for the channel 
 
        calc_cross_section
            Function to calculate the cross section
            for the channel 
        
        """

        self._light_product = light_product
        self._heavy_product = heavy_product
        self.A = light_product.A + heavy_product.A
        self.J = J
        self.pi = pi
        self.ell = ell
        self.ac = ac
        self.excitation = excitation 
        self.reduced_width_aplitudes = np.array(reduced_width_aplitudes)

    @abstractmethod
    def calc_k(self,incident_energies):
        """ Function to calculate k for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate k at

        Returns
        -------
        None
        
        """
        return None
    
    @abstractmethod
    def calc_rho(self,incident_energies):
        """ Function to calculate rho for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate rho at

        Returns
        -------
        None
        
        """
        return None
    
    @abstractmethod
    def calc_penetrability(self,incident_energies):
        """ Function to calculate the penetrability 
        for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate the
            penetrability at

        Returns
        -------
        None
        
        """
        return None
    
    @abstractmethod
    def calc_cross_section(self,U_matrix, k_sq, inc, out):
        """ Function to calculate the cross section
        for the channel 
        
        Parameters
        ----------
        U_matrix : numpy array
            The U-matrix for the compound nucleus

        k_sq : numpy array
            k values squared for the incident channel
            of the reaction

        inc : int
            The index of the incident channel in the U-matrix

        out : int
            The index of the outgoing channel in the U-matrix

        Returns
        -------
        None
        
        """
        return None
        

    def __repr__(self):
        return f'{self._light_product} + {self._heavy_product}({self.excitation/1e6} MeV)'
    
    def __str__(self):
        return f'{self._light_product} + {self._heavy_product}({self.excitation/1e6} MeV)'