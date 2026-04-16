import numpy as np
from   copy  import deepcopy
import pandas as pd
import os

from Approximations.models.math_models.R_matrix_new import rMatInterface
from Approximations.tools.gamma_tools               import reshape_RM
from   ATARI.ModelData.particle_pair                        import Particle_Pair
from   ATARI.ModelData.particle                             import Particle,          Neutron
from   ATARI.ModelData.experimental_model                   import Experimental_Model
from   ATARI.sammy_interface                                import sammy_classes,     template_creator, sammy_functions




class sammy():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Incident Nucleons"     in molecular_information,   "Model Gen Failed: Incident Nucleons not provided in molecular_information"
        assert "Target Nucleons"       in molecular_information,   "Model Gen Failed: Target Nucleons not provided in molecular_information"
        assert "Incident Spin"         in molecular_information,   "Model Gen Failed: Incident Spin not provided in molecular_information"
        assert "Target Spin"           in molecular_information,   "Model Gen Failed: Target Spin not provided in molecular_information"
        assert "Separation Energy"     in interaction_information, "Model Gen Failed: Separation Energy not provided in interaction_information"
        assert "Number Levels"         in interaction_information, "Model Gen Failed: Number Levels not provided in interaction_information"
        assert "Excited States"        in interaction_information, "Model Gen Failed: Excited States not provided in interaction_information"
        assert "Resonance Levels"      in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Effective Radius"      in interaction_information, "Model Gen Failed: Effective Radius not provided in interaction_information"
        assert "Elastic Variance"      in interaction_information, "Model Gen Failed: Elastic Variance not provided in interaction_information"
        assert "Capture Variance"      in interaction_information, "Model Gen Failed: Capture Variance not provided in interaction_information"
        assert "Capture Ell"           in interaction_information, "Model Gen Failed: Capture Ell not provided in interaction_information"
        assert "Gamma Matrix"          in interaction_information, "Model Gen Failed: Gamma Matrix not provided in interaction_information"
        assert "Energy Grid"           in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        assert "Data Format"           in model_information,       "Model Gen Failed: Data Format not provided in model_information"
        assert "Grid Reduction Factor" in model_information,      "Model Gen Failed: Grid Reduction Factor not provided in model_information"
        
        self.elastic_penetrability = interaction_information["Elastic Penetrability"]
        self.capture_penetrability = interaction_information["Capture Penetrability"]
        
        self.reduction_factor = model_information["Grid Reduction Factor"]
        
        self.energy_grid       = model_information["Energy Grid"][::self.reduction_factor]
        self.fudge_factor      = model_information["Fudge Factor"]
        
        target             = Particle(Z = int(molecular_information["Target Protons"]),
                              A = int(molecular_information["Target Nucleons"]),
                              mass = float(molecular_information["Target Nucleons"]),
                              name = molecular_information["Target Name"],
                              I = molecular_information["Target Spin"])

        spin = molecular_information["Incident Spin"] + molecular_information["Target Spin"]
        self.res_ladder         = pd.DataFrame({"E":       interaction_information["Resonance Levels"],
                                        "Gn1":     interaction_information["Elastic Widths"] * 1000,
                                        "Gg":      interaction_information["Capture Widths"] * 1000,
                                        "J_ID":    np.ones(interaction_information["Number Levels"]),
                                        "varyE":   np.ones(interaction_information["Number Levels"]),
                                        "varyGg":  np.ones(interaction_information["Number Levels"]),
                                        "varyGn1": np.ones(interaction_information["Number Levels"]),
                                        "varyGn2": np.zeros(interaction_information["Number Levels"]),
                                        "Jpi":     np.zeros(interaction_information["Number Levels"]) * spin,
                                        "L":       np.zeros(interaction_information["Number Levels"])})

        self.part_pair          = Particle_Pair(isotope          = molecular_information["Target Name"],
                                        resonance_ladder = self.res_ladder,
                                        formalism        = "XCT",
                                        ac               = interaction_information["Effective Radius"]/(10**-14),
                                        energy_range     = [np.min(self.energy_grid),np.max(self.energy_grid)],
                                        energy           = self.energy_grid,
                                        target           = target,
                                        projectile       = Neutron,
                                        l_max            = 1)

        self.part_pair.add_spin_group(Jpi     = str(spin),
                                      J_ID    = 1,
                                      D       = 9,
                                      gn2_avg = interaction_information["Elastic Variance"],
                                      gn2_dof = 1,
                                      gg2_avg = interaction_information["Capture Variance"],
                                      gg2_dof = 1)

        self.trans_model          = Experimental_Model(title        = "trans_model",
                                                       reaction     = "trans",
                                                       energy_grid  = self.energy_grid,
                                                       temp         = (interaction_information["Broadening Temperature"], 0))

        self.trans_model.template = os.path.realpath("sammy_template.inp")
        
        self.capture_model          = Experimental_Model(title        = "capture_model",
                                                         reaction     = "capture",
                                                         energy_grid  = self.energy_grid,
                                                         temp         = (interaction_information["Broadening Temperature"], 0))

        self.capture_model.template = os.path.realpath("sammy_template.inp")
        
        sammy_path         = os.path.expanduser("~")+"/.local/Sammy/bin/sammy"
        self.sammy_rto          = sammy_classes.SammyRunTimeOptions(sammy_path,
                                                            Print        = True,
                                                            bayes        = True,
                                                            keep_runDIR  = False,
                                                            sammy_runDIR = 'sammy_run_dir')
    
    def get_Initial_Guess(self, true_gamma_matrix):
        return(np.concat([true_gamma_matrix[:,0],np.diag(true_gamma_matrix[:,1:])]))
    
    def fit_call(self,fitting_parameters):
        def fit(initial_guess, data):
            trans_data   = pd.DataFrame({"E":self.energy_grid, "exp":data[::self.reduction_factor,0], "exp_unc":np.clip(data[::self.reduction_factor,0]*self.fudge_factor,1e-6,None)})
            capture_data = pd.DataFrame({"E":self.energy_grid, "exp":data[::self.reduction_factor,1], "exp_unc":np.clip(data[::self.reduction_factor,1]*self.fudge_factor,1e-6,None)})
            
            sammy_input = sammy_classes.SammyInputDataYW(particle_pair = self.part_pair,
                                                         resonance_ladder = self.res_ladder,  
                                                     
                                                         datasets= [trans_data, capture_data],
                                                         experiments = [self.trans_model, self.capture_model],
                                                         experimental_covariance= [{},{}], 
                                                         
                                                         max_steps = 10,
                                                         iterations = 3,
                                                         step_threshold = 0.01,
                                                         autoelim_threshold = None,
                                                     
                                                         LS = False,
                                                         LevMar = True,
                                                         LevMarV = 2,
                                                         LevMarVd= 5,
                                                         maxF= 1.5,
                                                         initial_parameter_uncertainty = 0.01)

            sammy_out = sammy_functions.run_sammy_YW(sammy_input, self.sammy_rto)
            
            elastic_widths = np.sqrt(np.abs(sammy_out.par_post["Gn1"].to_numpy())/(2000*self.elastic_penetrability))
            capture_widths = np.sqrt(np.abs(sammy_out.par_post["Gg"].to_numpy())/(2000*self.capture_penetrability))
            return(np.concat([elastic_widths,capture_widths]))
        return(fit)






# class ExtendedRMatrix(RMInterface):
#     def __init__(self, *args, **kwargs):
#         super(ExtendedRMatrix, self).__init__(*args, **kwargs)
#         self.init_guess_full={}



#     def evaluate(self,gamma_matrix:np.array,data:np.array)->float:
#         test_data = self.get_Data(gamma_matrix)
        
#         errors = np.zeros((self.num_channels, len(self.energy_grid)))
#         for idx in range(self.num_channels):
#             errors[idx,:] = data[1][idx] - test_data[1][idx]

#         total_error = np.sum(np.power(np.sum(errors[1:,:], 0), 2) + np.power(np.sum(errors, 0), 2))
#         total_error = np.sum(np.power(np.sum(errors[1:,:], 0), 2) + np.power(np.sum(errors, 0), 2))

#         return(total_error)



#     def calc_Gradient_And_Hessian(self,gamma_matrix,data,both=True):
#         data_types      = [float, complex]
#         energy_length   = self.energy_grid.size
#         num_independent = self.num_levels*2


#         #Creating a copy of the spin model to calculate values at the current position
#         test_data = self.get_Data(gamma_matrix)


#         #Calculating current element values
#         L        = self.spin_group.L_matrix
#         P        = self.spin_group.P_half
#         omega    = self.spin_group.omega_matrix
#         A_inv    = self.spin_group.A_matrix
#         U_matrix = self.spin_group.U_matrix
#         xs_fit   = np.zeros((self.num_channels,energy_length), data_types[0])
#         xs_true  = np.zeros((self.num_channels,energy_length), data_types[0])
#         for i in range(self.num_channels):
#             xs_fit[i]  = test_data[1][i]
#             xs_true[i] = data[1][i]
        

#         #Defines and sets up the gradients for the elements of the gamma matrix
#         gamma_gradient=np.zeros((num_independent,self.num_levels,self.num_channels),data_types[0])
#         for i in range(self.num_levels):
#             gamma_gradient[i,i,0]=1
#         for j in range(self.num_channels-1):
#             gamma_gradient[self.num_levels+j,j,j+1]=1
        

#         #Does work for the A Element
#         A_gradient=np.zeros((num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
#         for i in range(num_independent):
#             A_gradient[i]=-gamma_gradient[i]@L@gamma_matrix.T-gamma_matrix@L@gamma_gradient[i].T
        
#         if(both):
#             A_hessian=np.zeros((num_independent,num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
#             for i in range(num_independent):
#                 for j in range(num_independent):
#                     A_hessian[i,j]=-gamma_gradient[i]@L@gamma_gradient[j].T-gamma_gradient[j]@L@gamma_gradient[i].T


#         #Does work for the A Inverse Element
#         A_inv_gradient=np.zeros((num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
#         for i in range(num_independent):
#             A_inv_gradient[i]=-A_inv@A_gradient[i]@A_inv
        
#         if(both):
#             A_inv_hessian=np.zeros((num_independent,num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
#             for i in range(num_independent):
#                 for j in range(num_independent):
#                     A_inv_hessian[i,j]=A_inv@A_gradient[i]@A_inv@A_gradient[j]@A_inv-A_inv@A_hessian[i,j]@A_inv+A_inv@A_gradient[j]@A_inv@A_gradient[i]@A_inv


#         #Does work for the U Element
#         U_gradient=np.zeros((num_independent,energy_length,self.num_channels,self.num_channels),data_types[1])
#         for i in range(num_independent):
#             U_gradient[i]=2j*omega@P@(gamma_gradient[i].T@A_inv@gamma_matrix+
#                                     gamma_matrix.T@A_inv_gradient[i]@gamma_matrix+
#                                     gamma_matrix.T@A_inv@gamma_gradient[i])@P@omega
        
#         if(both):
#             U_hessian=np.zeros((num_independent,num_independent,energy_length,self.num_channels,self.num_channels),data_types[1])
#             for i in range(num_independent):
#                 for j in range(num_independent):
#                     interstage =gamma_gradient[i].T@A_inv_gradient[j]@gamma_matrix
#                     interstage+=gamma_gradient[i].T@A_inv@gamma_gradient[j]
#                     interstage+=gamma_gradient[j].T@A_inv_gradient[i]@gamma_matrix
#                     interstage+=gamma_matrix.T@A_inv_hessian[i,j]@gamma_matrix
#                     interstage+=gamma_matrix.T@A_inv_gradient[i]@gamma_gradient[j]
#                     interstage+=gamma_gradient[j].T@A_inv@gamma_gradient[i]
#                     interstage+=gamma_matrix.T@A_inv_gradient[j]@gamma_gradient[i]
#                     U_hessian[i,j]=2j*omega@P@interstage@P@omega


#         #Does work for the cross section
#         xs_gradient=np.zeros((num_independent,self.num_channels,energy_length),data_types[0])
#         for i in range(num_independent):
#             xs_gradient[i,0]=(10**24 * np.pi/self.spin_group.k_sq)*(-2*U_gradient[i,:,0,0].real+(np.conjugate(U_gradient[i,:,0,0])*U_matrix[:,0,0]+np.conjugate(U_matrix[:,0,0])*U_gradient[i,:,0,0]).real)
#             for j in range(1,self.num_channels):
#                 xs_gradient[i,j]=(10**24 * np.pi/self.spin_group.k_sq)*(np.conjugate(U_gradient[i,:,0,j])*U_matrix[:,0,j]+np.conjugate(U_matrix[:,0,j])*U_gradient[i,:,0,j]).real
        
#         if(both):
#             xs_hessian=np.zeros((num_independent,num_independent,self.num_channels,energy_length),data_types[0])
#             for i in range(num_independent):
#                 for j in range(num_independent):
#                     xs_hessian[i,j,0]=(10**24 * np.pi/self.spin_group.k_sq)*(-2*U_hessian[i,j,:,0,0].real+
#                                                                             (np.conjugate(U_hessian[i,j,:,0,0]) *U_matrix[:,0,0]+
#                                                                             np.conjugate(U_gradient[i,:,0,0])  *U_gradient[j,:,0,0]+
#                                                                             np.conjugate(U_gradient[j,:,0,0])  *U_gradient[i,:,0,0]+
#                                                                             np.conjugate(U_matrix[:,0,0])      *U_hessian[i,j,:,0,0]).real)
#                     for k in range(1,self.num_channels):
#                         xs_hessian[i,j,k]=(10**24 * np.pi/self.spin_group.k_sq)*(np.conjugate(U_hessian[i,j,:,0,k]) * U_matrix[:,0,k]+
#                                                                                 np.conjugate(U_gradient[i,:,0,k])  * U_gradient[j,:,0,k]+
#                                                                                 np.conjugate(U_gradient[j,:,0,k])  * U_gradient[i,:,0,k]+
#                                                                                 np.conjugate(U_matrix[:,0,k])      * U_hessian[i,j,:,0,k]).real


#         #Does work for the Error Term
#         error_gradient=np.zeros(num_independent,data_types[0])
#         for i in range(num_independent):
#             error_gradient[i]=np.sum(2*np.sum(xs_true-xs_fit,0)*np.sum(-xs_gradient[i],0)+2*np.sum(xs_true[1:]-xs_fit[1:],0)*np.sum(-xs_gradient[i,1:],0))
        
#         if(both):
#             error_hessian=np.zeros((num_independent,num_independent),data_types[0])
#             for i in range(num_independent):
#                 for j in range(num_independent):
#                     error_hessian[i,j]=np.sum(2*np.sum(-xs_gradient[j],0)      * np.sum(-xs_gradient[i],0)+
#                                             2*np.sum(xs_true-xs_fit,0)         * np.sum(-xs_hessian[i,j],0)+
#                                             2*np.sum(-xs_gradient[j,1:],0)     * np.sum(-xs_gradient[i,1:],0)+
#                                             2*np.sum(xs_true[1:]-xs_fit[1:],0) * np.sum(-xs_hessian[i,j,1:],0))
        
#         if(both):
#             return(error_gradient, error_hessian)
#         else:
#             return(error_gradient)