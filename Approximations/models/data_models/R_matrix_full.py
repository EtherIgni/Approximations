import numpy as np
from copy import deepcopy

from Approximations.models.math_models.R_matrix import RM_Interface






class R_Matrix_Full():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Incident Name"      in molecular_information,   "Model Gen Failed: Incident Name not provided in molecular_information"
        assert "Incident Protons"   in molecular_information,   "Model Gen Failed: Incident Protons not provided in molecular_information"
        assert "Incident Nucleons"  in molecular_information,   "Model Gen Failed: Incident Nucleons not provided in molecular_information"
        assert "departing Name"     in molecular_information,   "Model Gen Failed: departing Name not provided in molecular_information"
        assert "departing Protons"  in molecular_information,   "Model Gen Failed: departing Protons not provided in molecular_information"
        assert "departing Nucleons" in molecular_information,   "Model Gen Failed: departing Nucleons not provided in molecular_information"
        assert "Compound Name"      in molecular_information,   "Model Gen Failed: Compound Name not provided in molecular_information"
        assert "Compound Protons"   in molecular_information,   "Model Gen Failed: Compound Protons not provided in molecular_information"
        assert "Compound Nucleons"  in molecular_information,   "Model Gen Failed: Compound Nucleons not provided in molecular_information"
        assert "Separation Energy"  in interaction_information, "Model Gen Failed: Separation Energy not provided in interaction_information"
        assert "Number Levels"      in interaction_information, "Model Gen Failed: Number Levels not provided in interaction_information"
        assert "Excited States"     in interaction_information, "Model Gen Failed: Excited States not provided in interaction_information"
        assert "Resonance Levels"   in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Energy Grid"        in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        
        self.init_guess_full = {}
        excited_states       = interaction_information["Excited States"]
        resonances           = interaction_information["Resonance Levels"]
        energy_grid          = interaction_information["Energy Grid"]
        self.num_levels      = interaction_information["Number Levels"]
        self.num_channels    = len(excited_states)
        self.math_model      = RM_Interface(num_channels,self.num_levels)
    
        #Defines Interaction State
        incident  = Particle(molecular_information["Incident Name"],
                             molecular_information["Incident Protons"],
                             molecular_information["Incident Nucleons"])
        departing = Particle(molecular_information["departing Name"],
                             molecular_information["departing Protons"],
                             molecular_information["departing Nucleons"])
        target    = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Protons"],
                             molecular_information["Compound Nucleons"])
        compound  = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Protons"],
                             molecular_information["Compound Nucleons"],
                             Sn=interaction_information["Separation Energy"])
        self.math_model.set_incoming(incident)
        self.math_model.set_outgoing(departing)
        self.math_model.set_target(target)
        self.math_model.set_compound(compound)

        #Creates Elastic channel
        J                        = 3
        pi                       = 1
        ell                      = 0
        radius                   = 0.2
        reduced_width_amplitudes = np.ones(self.num_levels)
        self.math_model.set_elastic_channel(J,
                                            pi,
                                            ell,
                                            radius,
                                            reduced_width_amplitudes)

        #Creates Inelastic channel
        for idx,excitation in enumerate(excited_states):
            J                             = 3
            pi                            = 1
            ell                           = 0
            radius                        = 0.2
            reduced_width_amplitudes      = np.ones(self.num_levels)
            self.math_model.add_capture_channel(J,
                                                pi,
                                                ell,
                                                radius,
                                                reduced_width_amplitudes,
                                                excitation)

        self.math_model.set_resonance_energies(resonances)
        self.math_model.set_energy_grid(energy_grid)

        self.math_model.establish_spin_group()

        neutron_std       = np.sqrt(neutron_variance)
        gamma_matrix      = np.zeros((self.num_levels, self.num_channels))
        gamma_matrix[:,0] = np.random.normal(0,neutron_std, self.num_levels)
        for i in range(1, self.num_channels):
            running_sum = 0
            for excitation in excited_states:
                running_sum = running_sum+model.get_capture_channels()[i].calc_penetrability(separation_energy-excitation)
            gamma_std         = np.sqrt(gamma_variance/running_sum)
            gamma_matrix[:,i] = np.random.normal(0, gamma_std, number_levels)
        
        self.math_model.set_gamma_matrix(gamma_matrix)
    


    def generateData(self):
        