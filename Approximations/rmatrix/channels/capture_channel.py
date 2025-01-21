from Approximations.rmatrix.channels.abstract_channel import AbstractChannel
import numpy as np 

class CaptureChannel(AbstractChannel):
    def __init__(self,primary,product,J,pi,ell,ac,reduced_width_aplitudes,excitation):
        """ Class representing an elastic channel
        
        Parameters
        ----------
        primary : Particle object
            The primary gamma

        product : Particle object
            The product nucleus

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

        excitation : float
            the excitiation energy of the product nucleus in eV


        Attributes
        ----------
        primary : Particle object
            The primary gamma

        product : Particle object
            The product nucleus

        A : int
            Mass number of the product nucleus

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

        excitation : float
            the excitiation energy of the product nucleus in eV

        Sn : float
            the neutron separation energy of the product nucleus

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
        super().__init__(primary,product,J,pi,ell,ac,reduced_width_aplitudes, excitation)
        self.Sn = product.Sn

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
        hbar = 6.582119e-16  # eV-s
        c = 2.99792e10  # cm/s
        e_gamma = self.Sn + np.array(incident_energies) - self.excitation
        k_cm = e_gamma / (hbar * c)
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
        return (self.calc_k(incident_energies) * self.ac*10**(-12))**(2*self.ell + 1)

    def calc_cross_section(self,U_matrix,k_sq, inc, out):
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
            the cross section values
        
        """
        self.cross_section = 10**24 * np.pi/k_sq * np.conjugate(U_matrix[:,inc,out])*U_matrix[:,inc,out] 
        
        # check that the imaginay component is basically zero before dropping it
        max_ind = np.argmax(self.cross_section.imag)
        #assert self.cross_section[max_ind].imag / self.cross_section[max_ind].real < 1e-10
        self.cross_section = self.cross_section.real