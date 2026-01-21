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
        assert "Gamma Matrix"       in interaction_information, "Model Gen Failed: Gamma Matrix not provided in interaction_information"
        assert "Energy Grid"        in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        assert "Data Format"        in model_information,       "Model Gen Failed: Data Format not provided in model_information"
        
        self.init_guess   = {}
        self.data_format  = model_information["Data Format"]
        excited_states    = interaction_information["Excited States"]
        resonances        = interaction_information["Resonance Levels"]
        energy_grid       = model_information["Energy Grid"]
        self.num_levels   = interaction_information["Number Levels"]
        self.num_channels = len(excited_states)+1
        self.mathModel    = RMInterface(self.num_levels, self.num_channels)

        self.mathModel.set_Molecular_Information(molecular_information["Incident Nucleons"],
                                                 molecular_information["Target Nucleons"],
                                                 interaction_information["Separation Energy"])
        
        self.mathModel.set_Energy_Grid(energy_grid)
        
        self.mathModel.set_Resonance_Energy(resonances)
        
        self.mathModel.set_Excitation_Energy(excited_states)
        
        self.mathModel.set_Elastic_information(interaction_information["Elastic Variance"],
                                               interaction_information["Elastic Radius"])
        
        self.mathModel.set_Capture_Information(interaction_information["Capture Variance"],
                                               interaction_information["Capture Radius"],
                                               interaction_information["Capture Ell"])
        
        self.mathModel.set_Gamma_Matrix(interaction_information["Gamma Matrix"])
        
        self.mathModel.calc_Cross_Sections()
        
        self.true_gamma_matrix = self.mathModel.get_Gamma_Matrix()
        self.energy_grid       = self.mathModel.energy_grid
    


    def get_Cross_Sections(self):
        return(self.mathModel.XS)
    
    def generate_Data(self):
        cross_sections = self.get_Cross_Sections()
        
        if(self.data_format == "Total and Gamma"):
            measures       = np.zeros((self.mathModel.energy_length, 2))
            measures[:,0]  = np.sum(cross_sections,1)
            measures[:,1]  = np.sum(cross_sections[:,1:],1)
        if(self.data_format == "Full"):
            measures = cross_sections
        else:
            measures = np.sum(cross_sections,1)
        
        return(measures)