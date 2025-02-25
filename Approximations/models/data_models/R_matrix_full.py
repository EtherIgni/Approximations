import numpy as np
from   copy  import deepcopy

from Approximations.models.math_models.R_matrix import RMInterface
from Approximations.rmatrix.base.particles      import Particle






class RMatrixFull():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Incident Name"      in molecular_information,   "Model Gen Failed: Incident Name not provided in molecular_information"
        assert "Incident Protons"   in molecular_information,   "Model Gen Failed: Incident Protons not provided in molecular_information"
        assert "Incident Nucleons"  in molecular_information,   "Model Gen Failed: Incident Nucleons not provided in molecular_information"
        assert "Departing Name"     in molecular_information,   "Model Gen Failed: Departing Name not provided in molecular_information"
        assert "Departing Protons"  in molecular_information,   "Model Gen Failed: Departing Protons not provided in molecular_information"
        assert "Departing Nucleons" in molecular_information,   "Model Gen Failed: Departing Nucleons not provided in molecular_information"
        assert "Compound Name"      in molecular_information,   "Model Gen Failed: Compound Name not provided in molecular_information"
        assert "Compound Protons"   in molecular_information,   "Model Gen Failed: Compound Protons not provided in molecular_information"
        assert "Compound Nucleons"  in molecular_information,   "Model Gen Failed: Compound Nucleons not provided in molecular_information"
        assert "Separation Energy"  in interaction_information, "Model Gen Failed: Separation Energy not provided in interaction_information"
        assert "Number Levels"      in interaction_information, "Model Gen Failed: Number Levels not provided in interaction_information"
        assert "Excited States"     in interaction_information, "Model Gen Failed: Excited States not provided in interaction_information"
        assert "Resonance Levels"   in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Neutron Variance"   in interaction_information, "Model Gen Failed: Neutron Variance not provided in interaction_information"
        assert "Gamma Variance"     in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Energy Grid"        in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        
        self.init_guess_full = {}
        excited_states       = interaction_information["Excited States"]
        resonances           = interaction_information["Resonance Levels"]
        energy_grid          = model_information["Energy Grid"]
        self.num_levels      = interaction_information["Number Levels"]
        self.num_channels    = len(excited_states)+1
        self.math_model      = RMInterface(self.num_channels,self.num_levels)
    
        #Defines Interaction State
        incident  = Particle(molecular_information["Incident Name"],
                             molecular_information["Incident Nucleons"],
                             molecular_information["Incident Protons"])
        departing = Particle(molecular_information["Departing Name"],
                             molecular_information["Departing Nucleons"],
                             molecular_information["Departing Protons"])
        target    = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Nucleons"],
                             molecular_information["Compound Protons"])
        compound  = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Nucleons"],
                             molecular_information["Compound Protons"],
                             Sn=interaction_information["Separation Energy"])
        self.math_model.set_Incoming(incident)
        self.math_model.set_Outgoing(departing)
        self.math_model.set_Target(target)
        self.math_model.set_Compound(compound)

        #Creates Elastic channel
        J                        = 3
        pi                       = 1
        ell                      = 0
        radius                   = 0.2
        reduced_width_amplitudes = np.ones(self.num_levels)
        self.math_model.set_Elastic_Channel(J,
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
            self.math_model.add_Capture_Channel(J,
                                                pi,
                                                ell,
                                                radius,
                                                reduced_width_amplitudes,
                                                excitation)

        self.math_model.set_Resonance_Energies(resonances)
        self.math_model.set_Energy_Grid(energy_grid)

        self.math_model.establish_Spin_Group()

        neutron_std       = np.sqrt(interaction_information["Neutron Variance"])
        gamma_matrix      = np.zeros((self.num_levels, self.num_channels))
        gamma_matrix[:,0] = np.random.normal(0,neutron_std, self.num_levels)
        for i in range(self.num_channels-1):
            running_sum = 0
            for excitation in excited_states:
                running_sum = running_sum+self.math_model.get_Capture_Channels()[i].calc_penetrability(interaction_information["Separation Energy"]-excitation)
            gamma_std         = np.sqrt(interaction_information["Gamma Variance"]/running_sum)
            gamma_matrix[:,i+1] = np.random.normal(0, gamma_std, self.num_levels)
        
        self.math_model.set_Gamma_Matrix(gamma_matrix)
    


    def generate_Data(self):
        return(self.math_model.get_Data())