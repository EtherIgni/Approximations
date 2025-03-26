import numpy as np

from Approximations.models.math_models.R_matrix_new import RMInterface






class RMatrixFull():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Incident Nucleons"  in molecular_information,   "Model Gen Failed: Incident Nucleons not provided in molecular_information"
        assert "Target Nucleons"    in molecular_information,   "Model Gen Failed: Target Nucleons not provided in molecular_information"
        assert "Separation Energy"  in interaction_information, "Model Gen Failed: Separation Energy not provided in interaction_information"
        assert "Number Levels"      in interaction_information, "Model Gen Failed: Number Levels not provided in interaction_information"
        assert "Excited States"     in interaction_information, "Model Gen Failed: Excited States not provided in interaction_information"
        assert "Resonance Levels"   in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Elastic Variance"   in interaction_information, "Model Gen Failed: Elastic Variance not provided in interaction_information"
        assert "Elastic Radius"     in interaction_information, "Model Gen Failed: Elastic Radius not provided in interaction_information"
        assert "Capture Variance"   in interaction_information, "Model Gen Failed: Capture Variance not provided in interaction_information"
        assert "Capture Radius"     in interaction_information, "Model Gen Failed: Capture Radius not provided in interaction_information"
        assert "Capture Ell"        in interaction_information, "Model Gen Failed: Capture Ell not provided in interaction_information"
        assert "Energy Grid"        in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        
        self.init_guess   = {}
        excited_states    = interaction_information["Excited States"]
        resonances        = interaction_information["Resonance Levels"]
        energy_grid       = model_information["Energy Grid"]
        self.num_levels   = interaction_information["Number Levels"]
        self.num_channels = len(excited_states)+1
        self.math_model   = RMInterface(self.num_levels, self.num_channels)

        self.math_model.set_Molecular_Information(molecular_information["Incident Nucleons"],
                                                  molecular_information["Target Nucleons"],
                                                  interaction_information["Separation Energy"])
        
        self.math_model.set_Energy_Grid(energy_grid)
        
        self.math_model.set_Resonance_Energy(resonances)
        
        self.math_model.set_Excitation_Energy(excited_states)
        
        self.math_model.set_Elastic_information(interaction_information["Elastic Variance"],
                                                interaction_information["Elastic Radius"])
        
        self.math_model.set_Capture_Information(interaction_information["Capture Variance"],
                                                interaction_information["Capture Radius"],
                                                interaction_information["Capture Ell"])
        
        self.math_model.generate_Gamma_Matrix()
        
        self.math_model.calc_Cross_Sections()
    


    def generate_Data(self):
        measures      = np.zeros((self.math_model.energy_length, 2))
        measures[:,0] = np.sum(self.math_model.XS,1)
        measures[:,1] = np.sum(self.math_model.XS[:,1:],1)
        return(measures)