import numpy as np

class RMInterface():
    hbar             = 6.582119e-16  # eV-s
    c                = 2.99792e10    # cm/s
    ρ_scale          = 1e-12
    elastic_const    = 0.002197e12
    
    data_types       = [np.float64,
                        np.complex64]
    
    
    
    def __init__(self, num_levels, num_channels):
        self.num_levels   = num_levels
        self.num_channels = num_channels
        
        self.compound_nucleons  = None
        self.separation_energy  = None
        self.energy_grid        = None
        self.energy_length      = None
        self.resonance_energies = None
        self.elastic_variance   = None
        self.elastic_radius     = None
        self.capture_variance   = None
        self.capture_radius     = None
        self.capture_ell        = None



    def set_Molecular_Information(self, incoming_nucleons, target_nucleons, separation_energy):
        self.compound_nucleons = incoming_nucleons + target_nucleons
        self.separation_energy = separation_energy



    def set_Energy_Grid(self, energy_grid):
        self.energy_grid   = energy_grid
        self.energy_length = len(energy_grid)



    def set_Resonance_Energy(self, resonance_energies):
        self.resonance_energies = resonance_energies



    def set_Excitation_Energy(self, excitation_energies):
        self.excitation_energies = excitation_energies



    def set_Elastic_information(self, variance, radius):
        self.elastic_variance = variance
        self.elastic_radius   = radius



    def set_Capture_Information(self, variance, radius, ell):
        self.capture_variance = variance
        self.capture_radius   = radius
        self.capture_ell      = ell



    def set_Gamma_Matrix(self,
                         gamma_matrix):
        self.gamma_matrix       = gamma_matrix.astype(self.data_types[0])
        
    def set_Gamma_Matrix(self, gamma_matrix):
        self.gamma_matrix = gamma_matrix
        self.gamma_matrix = self.gamma_matrix.astype(self.data_types[0])
        
    def get_Gamma_Matrix(self):
        return(self.gamma_matrix)



    def calc_Cross_Sections(self):
        k            = np.zeros((self.energy_length, self.num_channels, self.num_channels))
        k[:,0,0]     = (self.elastic_const*self.compound_nucleons*np.sqrt(self.energy_grid))/(self.compound_nucleons+1)
        for i in range(1,self.num_channels):
            k[:,i,i] = (self.separation_energy+self.energy_grid-self.excitation_energies[i-1])/(self.hbar*self.c)
        k            = k.astype(self.data_types[0])

        ρ = k*self.elastic_radius*self.ρ_scale
        ρ = ρ.astype(self.data_types[0])

        P          = np.zeros((self.energy_length, self.num_channels, self.num_channels))
        P[:,0,0]   = ρ[:,0,0]
        P[:,1:,1:] = np.power(ρ[:,1:,1:],2*self.capture_ell+1)
        P          = P.astype(self.data_types[1])

        L = 1j*P

        P_root = np.sqrt(P)

        Γ = np.repeat(self.gamma_matrix[np.newaxis,:,:], self.energy_length, 0)

        O = Γ@L@np.transpose(Γ, [0,2,1])

        E1 = np.diag(self.resonance_energies)
        E1 = np.repeat(E1[np.newaxis,:,:], self.energy_length, 0)
        E2 = np.eye(self.num_levels)
        E2 = np.repeat(E2[np.newaxis,:,:], self.energy_length, 0)
        E2 = self.energy_grid.reshape(self.energy_length, 1, 1)*E2
        E  = E1-E2
        E  = E.astype(self.data_types[1])

        A = E-O

        A_inv = np.linalg.inv(A)

        Q = np.transpose(Γ, [0,2,1])@A_inv@Γ

        I = np.eye(self.num_channels)
        I = np.repeat(I[np.newaxis,:,:], self.energy_length, 0)
        I = I.astype(self.data_types[1])

        W = I + 2j*P_root@Q@P_root

        Ω = I[0]*np.repeat(np.exp(-1j*np.diagonal(ρ,axis1=1,axis2=2))[:,:,np.newaxis],self.num_channels,2)
        Ω = Ω.astype(self.data_types[1])

        U = Ω@W@Ω

        k2 = np.power(k[:,0,0],2)
        
        self.XS            = np.zeros((self.energy_length, self.num_channels), self.data_types[0])
        self.XS[:,0]     = (10**24 * np.pi/k2 * (1- 2*U[:,0,0].real + np.conjugate(U[:,0,0])*U[:,0,0])).real
        for i in range(1, self.num_channels):
            self.XS[:,i] = (10**24 * np.pi/k2 * np.conjugate(U[:,0,i])*U[:,0,i]).real