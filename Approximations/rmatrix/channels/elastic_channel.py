from rmatrix.channels.abstract_channel import AbstractChannel
import numpy as np
import sys

class ElasticChannel(AbstractChannel):
    def __init__(self,neutron,target,J,pi,ell,ac,reduced_width_aplitudes):

        """ Class representing an elastic channel
        
        Parameters
        ----------
        neutron : Particle object
            The light product in the channel

        target : Particle object
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


        Attributes
        ----------
        neutron : Particle object
            The light product in the channel

        target : Particle object
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
        if ell != 0:
            sys.exit("Only set up for s-wave neutrons right now")

        super().__init__(neutron,target,J,pi,ell,ac,reduced_width_aplitudes)

    def calc_k(self,incident_energies):
        """ Function to calculate k for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate k at

        Returns
        -------
        np.array
            the k values
        
        """
        const = 0.002197*10**(12)    # c per sqrt(eV)
        k_cm = const*self.A*np.sqrt(incident_energies)/(self.A+1) # 1/cm^1/2
        return k_cm

    def calc_rho(self,incident_energies):
        """ Function to calculate rho for the channel 
        
        Parameters
        ----------
        incident_energies : list
            Incident energies in eV to calculate rho at

        Returns
        -------
        np.array
            the rho values
        
        """
        k_cm = self.calc_k(incident_energies)
        ac_cm = self.ac*10**(-12)
        return k_cm*ac_cm
        
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
        np.array
            the penetrability values
        
        """
        if self.ell == 0:
            return self.calc_rho(incident_energies)
        

    def calc_cross_section(self, U_matrix, k_sq, inc, out):
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
        np.array
            the cross section
        
        """
        self.cross_section = 10**24 * np.pi/k_sq * (1- 2*U_matrix[:,inc,out].real + np.conjugate(U_matrix[:,inc,out])*U_matrix[:,inc,out])

        # check that the imaginay component is basically zero before dropping it
        max_ind = np.argmax(self.cross_section.imag / self.cross_section.real)
        #assert self.cross_section[max_ind].imag / self.cross_section[max_ind].real < 1e-10

        self.cross_section = self.cross_section.real