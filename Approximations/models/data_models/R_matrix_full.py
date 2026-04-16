import numpy as np



from Approximations.models.math_models.R_matrix_new import rMatInterface






class rMatrixFull():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Incident Nucleons"       in molecular_information,   "Model Gen Failed: Incident Nucleons not provided in molecular_information"
        assert "Target Nucleons"         in molecular_information,   "Model Gen Failed: Target Nucleons not provided in molecular_information"
        assert "Incident Spin"           in molecular_information,   "Model Gen Failed: Incident Spin not provided in molecular_information"
        assert "Target Spin"             in molecular_information,   "Model Gen Failed: Target Spin not provided in molecular_information"
        assert "Separation Energy"       in interaction_information, "Model Gen Failed: Separation Energy not provided in interaction_information"
        assert "Number Levels"           in interaction_information, "Model Gen Failed: Number Levels not provided in interaction_information"
        assert "Excited States"          in interaction_information, "Model Gen Failed: Excited States not provided in interaction_information"
        assert "Resonance Levels"        in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Effective Radius"        in interaction_information, "Model Gen Failed: Effective Radius not provided in interaction_information"
        assert "Elastic Variance"        in interaction_information, "Model Gen Failed: Elastic Variance not provided in interaction_information"
        assert "Capture Variance"        in interaction_information, "Model Gen Failed: Capture Variance not provided in interaction_information"
        assert "Capture Ell"             in interaction_information, "Model Gen Failed: Capture Ell not provided in interaction_information"
        assert "Gamma Matrix"            in interaction_information, "Model Gen Failed: Gamma Matrix not provided in interaction_information"
        assert "Broaden Cross Section"   in interaction_information, "Model Gen Failed: Broaden Cross Section not provided in interaction_information"
        assert "Convert to Transmission" in interaction_information, "Model Gen Failed: Convert to Transmission not provided in interaction_information"
        assert "Energy Grid"             in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        assert "Data Format"             in model_information,       "Model Gen Failed: Data Format not provided in model_information"
        if(interaction_information["Broaden Cross Section"]):
            assert "Broadening Temperature" in interaction_information, "Model Gen Failed: Broadening Temperature not provided in interaction_information"
        if(interaction_information["Convert to Transmission"]):
            assert "Material Thickness" in interaction_information, "Model Gen Failed: Material Thickness not provided in interaction_information"
        
        self.init_guess   = {}
        self.data_format  = model_information["Data Format"]
        excited_states    = interaction_information["Excited States"]
        resonances        = interaction_information["Resonance Levels"]
        energy_grid       = model_information["Energy Grid"]
        self.num_levels   = interaction_information["Number Levels"]
        self.num_channels = len(excited_states)+1
        self.mathModel    = rMatInterface(self.num_levels, self.num_channels)

        self.mathModel.set_Molecular_Information(molecular_information["Incident Nucleons"],
                                                 molecular_information["Target Nucleons"],
                                                 molecular_information["Incident Spin"],
                                                 molecular_information["Target Spin"],
                                                 interaction_information["Effective Radius"],
                                                 interaction_information["Separation Energy"])
        
        self.mathModel.set_Energy_Grid(energy_grid)
        
        self.mathModel.set_Resonance_Energy(resonances)
        
        self.mathModel.set_Excitation_Energy(excited_states)
        
        self.mathModel.set_Elastic_information(interaction_information["Elastic Variance"])
        
        self.mathModel.set_Capture_Information(interaction_information["Capture Variance"],
                                               interaction_information["Capture Ell"])
        
        self.mathModel.set_Gamma_Matrix(interaction_information["Gamma Matrix"])
        
        self.mathModel.calc_Cross_Sections()
        if(interaction_information["Broaden Cross Section"]):
            self.mathModel.doppler_Broaden(interaction_information["Broadening Temperature"])
        if(interaction_information["Convert to Transmission"]):
            self.mathModel.convert_Total_To_Transmission(interaction_information["Material Thickness"])
        
        self.true_gamma_matrix = self.mathModel.get_Gamma_Matrix()
        self.energy_grid       = self.mathModel.energy_grid
        
        resonance_penetrabilies = self.mathModel.calc_Penetrability(np.array(interaction_information["Resonance Levels"]))
        interaction_information["Elastic Widths"] = 2 * np.power(interaction_information["Gamma Matrix"][:,0],2) * resonance_penetrabilies[:,0]
        interaction_information["Capture Widths"] = 2 * np.sum(np.power(interaction_information["Gamma Matrix"][:,1:],2) * resonance_penetrabilies[:,1:],1)
        interaction_information["Elastic Penetrability"] = resonance_penetrabilies[:,0]
        interaction_information["Capture Penetrability"] = np.sum(resonance_penetrabilies[:,1:],1)
    
    def generate_Data(self):
        cross_sections = self.mathModel.cross_section
        
        return(cross_sections)