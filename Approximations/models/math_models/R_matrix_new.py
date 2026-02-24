import numpy as np

class rMatInterface():
    h_bar        = 6.582119e-16    # eV-s
    c            = 2.99792e8       # m/s
    neutron_mass = 9.3956e8        # eV
    barns        = 10**28          # barns/m^2
    boltz        = 8.617333262e-5  # eV/K
    
    
    
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



    def set_Molecular_Information(self, incident_nucleons, target_nucleons, incident_spin, target_spin, effective_radius, separation_energy):
        self.incident_nucleons = incident_nucleons
        self.target_nucleons   = target_nucleons
        self.incident_spin     = incident_spin
        self.target_spin       = target_spin
        self.effective_radius  = effective_radius
        self.separation_energy = separation_energy



    def set_Energy_Grid(self, energy_grid):
        self.energy_grid   = energy_grid
        self.energy_length = len(energy_grid)



    def set_Resonance_Energy(self, resonance_energies):
        self.resonance_energies = resonance_energies



    def set_Excitation_Energy(self, excitation_energies):
        self.excitation_energies = excitation_energies

    def set_Elastic_information(self, variance):
        self.elastic_variance = variance



    def set_Capture_Information(self, variance, ell):
        self.capture_variance = variance
        self.capture_ell      = ell



    def set_Gamma_Matrix(self,
                         gamma_matrix):
        self.gamma_matrix       = gamma_matrix
        
    def get_Gamma_Matrix(self):
        return(self.gamma_matrix)



    def calc_Penetrability(self, energy):
        momentum        = np.zeros((energy.size, self.num_channels))
        momentum[:,0] = (1/(self.h_bar*self.c)) * np.sqrt(((2 * self.neutron_mass * self.incident_nucleons * self.target_nucleons) * energy) / (self.incident_nucleons + self.target_nucleons))
        for i in range(1,self.num_channels):
            momentum[:,i] = (energy + self.separation_energy - self.excitation_energies[i-1]) / (self.h_bar*self.c)
        
        penetrability       = momentum * self.effective_radius
        penetrability[:,1:] = np.power(penetrability[:,1:], 2*self.capture_ell+1)
        
        return(penetrability)

    def calc_Cross_Sections(self):
        def extend(matrix):
            return(np.repeat(matrix[np.newaxis,:,:], self.energy_length, 0))
        
        self.penetrability = self.calc_Penetrability(self.energy_grid)

        L_factor      = np.eye(self.num_channels) * (1j * self.penetrability)[:,:,None]
        P_half_factor = np.eye(self.num_channels) * np.sqrt(self.penetrability)[:,:,None]
        Ω_factor      = np.eye(self.num_channels) * np.exp(-1j * self.penetrability)[:,:,None]
        
        gamma_matrix  = np.repeat(self.gamma_matrix[np.newaxis,:,:], self.energy_length, 0)

        E_factor = extend(np.diag(self.resonance_energies)) - extend(np.eye(self.num_levels)) * self.energy_grid[:,None,None]

        A_inv = E_factor - gamma_matrix @ L_factor @ np.transpose(gamma_matrix, [0,2,1])

        A_matrix = np.linalg.inv(A_inv)

        W_matrix = extend(np.eye(self.num_channels)) + 2j*(P_half_factor @ np.transpose(gamma_matrix, [0,2,1]) @ A_matrix @ gamma_matrix @ P_half_factor)

        U_matrix = Ω_factor @ W_matrix @ Ω_factor
        

        neutron_momentum = self.penetrability[:,0]/self.effective_radius
        spin_factor      = (2*(self.incident_spin+self.target_spin)+1)/((2*self.incident_spin+1)*(2*self.target_spin+1))
        norm_const       = self.barns * np.pi * spin_factor * np.power(neutron_momentum,-2)
        
        self.cross_section      = np.zeros((self.energy_length, 2))
        elastic                 = norm_const * (1 - 2*U_matrix[:,0,0].real + U_matrix[:,0,0]*np.conjugate(U_matrix[:,0,0])).real
        self.cross_section[:,1] = norm_const * np.sum(U_matrix[:,0,1:]*np.conjugate(U_matrix[:,0,1:]),1).real
        self.cross_section[:,0] = elastic + self.cross_section[:,1]
        
    def doppler_Broaden(self, temperature):
        doppler_width = np.sqrt(4*self.boltz*self.incident_nucleons*temperature*self.energy_grid[:,None]/self.target_nucleons)
        
        energy_factor = np.sqrt(self.energy_grid[None,:]*self.energy_grid[:,None])
        first_exponential  = np.exp(-4*np.power(self.energy_grid[:,None]-energy_factor,2)/np.power(doppler_width,2))
        second_exponential = np.exp(-4*np.power(self.energy_grid[:,None]+energy_factor,2)/np.power(doppler_width,2))
        kernel = (first_exponential-second_exponential) * np.sqrt(self.energy_grid[:,None]/self.energy_grid[None,:])
        integrand = kernel[:,:,None] * self.cross_section[None,:,:]
        integral  = np.trapezoid(integrand,self.energy_grid[None,:,None],axis=1)
        
        self.cross_section = integral/(doppler_width[:,0, None]*np.sqrt(np.pi))