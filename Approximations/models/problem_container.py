from Approximations.models.data_models.R_matrix_full import R_Matrix_Full
from Approximations.models.fitting_models.reich_moore_model import Reich_Moore
from Approximations.models.fitting_models.gamma_SVD_model import Gamma_SVD

class problem():
    data_model_formats= [R_Matrix_Full]
    fit_model_formats=  [Reich_Moore,
                         Gamma_SVD]

    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information,
                 selections):
        # molecular_information:   Incident Name
        #                          Compound Name
        #                          Num Protons
        #                          Num Nucleons
        #
        # interaction_information: Separation Energy
        #                          Resonance Distance
        #                          Resonance Average Spacing
        #                          Number Levels
        #                          Excited States
        #                          Gamma Variance
        #                          Neutron Variance
        #
        # model_information:       Energy Grid Range
        #                          Energy Grid Buffer
        #
        # Fitting Parameters:      Max Iterations
        #
        # selections:              Data Model
        #                              1: Complete R-Matrix
        #                          Fit Model
        #                              1: Reich Moore
        #                              2: SVD
        #                          Fit Method
        #                              1: Built in gradient descent
        #                              2: Built in Levenburg-Maquartself.

        self.molecular_information=molecular_information
        self.interaction_information=interaction_information
        self.model_information=model_information
        self.selections=selections

        self.data_model=self.form_model(self.data_model_formats,
                                        self.selections["Data Model"])
        self.fit_model=self.form_model(self.data_model_formats,
                                       self.selections["Fit Model"])

        self.fit_call=self.form_fit_call()

    def form_model(self,
                   model_formats,
                   model_selection):
        model=model_formats[model_selection-1]
        return(model(self.molecular_information,
                     self.interaction_information,
                     self.model_information))
    
    def form_fit_call(self):
        pass

    def generate_data(self):
        self.data=self.data_model.generate_data()