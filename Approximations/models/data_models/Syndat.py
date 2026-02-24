import os
import contextlib
import numpy      as np
import pandas     as pd


from   Approximations.models.data_models.R_matrix_full      import rMatrixFull
from   ATARI.ModelData.particle_pair                        import Particle_Pair
from   ATARI.ModelData.particle                             import Particle,          Neutron
from   ATARI.ModelData.experimental_model                   import Experimental_Model
from   ATARI.syndat.data_classes                            import syndatOPT
from   ATARI.syndat.syndat_model                            import Syndat_Model
from   ATARI.sammy_interface                                import sammy_classes,     template_creator
from   ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI



sammy_path            = "/home/aaron/Depo/SAMMY/sammy/build/bin/sammy"
capture_template_path = os.getcwd() + "/Approximations/syndat_templates/capture_template.inp"





class syndatData():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Target Protons"     in molecular_information,   "Model Gen Failed: Target Protons not provided in molecular_information"
        assert "Target Nucleons"    in molecular_information,   "Model Gen Failed: Target Nucleons not provided in molecular_information"
        assert "Target Name"        in molecular_information,   "Model Gen Failed: Target Name not provided in molecular_information"
        assert "Number Levels"      in interaction_information, "Model Gen Failed: Number Levels not provided in interaction_information"
        assert "Resonance Levels"   in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Elastic Variance"   in interaction_information, "Model Gen Failed: Elastic Variance not provided in interaction_information"
        assert "Capture Variance"   in interaction_information, "Model Gen Failed: Capture Variance not provided in interaction_information"
        assert "Gamma Matrix"       in interaction_information, "Model Gen Failed: Gamma Matrix not provided in interaction_information"
        assert "Energy Grid"        in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        
        self.num_levels   = interaction_information["Number Levels"]
        
        self.rawDataModel = rMatrixFull(molecular_information,
                                        interaction_information,
                                        model_information)
        self.true_gamma_matrix = self.rawDataModel.true_gamma_matrix
        
        self.energy_grid       = model_information["Energy Grid"]
        
        self.exp_model          = Experimental_Model(title        = "sim_model",
                                                     reaction     = "capture",
                                                     energy_grid  = self.energy_grid,
                                                     temp         = (interaction_information["Broadening Temperature"], 0))

        self.exp_model.template = os.path.realpath("capture_template.inp")
        
    def generate_Data(self):
        raw_data = self.rawDataModel.generate_Data()
        
        data_composite = pd.DataFrame({"E":self.energy_grid, "true":raw_data})

        syn_opt   = syndatOPT(sampleRES            = False,
                              calculate_covariance = False,
                              explicit_covariance  = False)

        with contextlib.redirect_stdout(None):
            capture_gen_model = Capture_Yield_RPI()
            capture_gen_model.approximate_unknown_data(exp_model  = self.exp_model,
                                                    smooth     = False,
                                                    check_trig = True)
            capture_red_model = Capture_Yield_RPI()
            capture_red_model.approximate_unknown_data(exp_model  = self.exp_model,
                                                    smooth     = False,
                                                    check_trig = True)

        syn_model = Syndat_Model(self.exp_model,
                                 capture_gen_model,
                                 capture_red_model,
                                 options = syn_opt)
        syn_model.sample(num_samples = 1,
                         pw_true     = data_composite)

        data_sample  = syn_model.samples[0]
        syndat_dataframe = data_sample.pw_reduced[["E","exp","exp_unc"]]

        return(syndat_dataframe)