import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from Approximations.models.problem_container import Problem

from   ATARI.ModelData.particle_pair                        import Particle_Pair
from   ATARI.ModelData.particle                             import Particle,          Neutron
from   ATARI.ModelData.experimental_model                   import Experimental_Model
from   ATARI.syndat.data_classes                            import syndatOPT
from   ATARI.syndat.syndat_model                            import Syndat_Model
from   ATARI.sammy_interface                                import sammy_classes,     template_creator, sammy_functions
from   ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI






molecular_information   = {"Incident Name":      "n",
                           "Incident Nucleons":  1,
                           "Incident Protons":   0,
                           "Incident Spin":      0.5,
                           "Departing Name":     "g",
                           "Departing Nucleons": 0,
                           "Departing Protons":  0,
                           "Target Name":        "Ta181",
                           "Target Nucleons":    181,
                           "Target Protons":     71,
                           "Target Spin":        3.5}

interaction_information = {"Separation Energy":         float(7.5767E6),
                           "Effective Radius":          float(8.127e-15),
                           "Elastic Variance":          float(452.5E-3),
                           "Capture Variance":          float(32E-3),
                           "Capture Ell":               0,
                           "Number Levels":             2,
                           "Resonance Distance":        200,
                           "Resonance Average Spacing": 9,
                           "Resonance Levels":          np.array([215, 235]),
                        #    "Excitation Model":          "Bimodal",
                        #    "Excitation Limits":         np.array([1E3,6E6,7E6]),
                        #    "Number Excitation States":  6,
                           "Excited States":            np.array([0,1000]),
                        #    "Gamma Matrix":              np.array([[ 0.39962476, 0.0118576 ,  0.02952645], [0.13179563, 0.01365162, 0.02872839]]),
                           "Gamma Matrix":              np.array([[ 0.39962476, 0.0118576 ,  0], [0.13179563, 0, 0.02872839]]),
                           "Broadening Temperature":    300,
                           "Broaden Cross Section":     True,
                           "Material Thickness":        0.067166,
                           "Convert to Transmission":   True}

model_information       = {"Energy Grid Gen Type":            "Uniform",
                           "Energy Grid Size":                6000,
                           "Energy Grid Buffer":              5,
                           "Grid Reduction Factor":           6,
                           "Data Format":                     "Full",
                           "Fudge Factor":                    0.01}

fitting_parameters      = {"Iteration Limit":        100,
                           "Improvement Threshold":  0.1,
                           "Initial Priority":       float(10E6),
                           "Priority Multiplier":    1.5,
                           "Priority Minimum":       float(10E-8),
                           "Priority Maximum":       float(10E16)}

selections              = {"Data Model": 1,
                           "Fit Model":  2,
                           "Fit Method": 1}

problem      = Problem(molecular_information,
                       interaction_information,
                       model_information,
                       fitting_parameters,
                       selections)

energy_grid  = problem.model_information["Energy Grid"]
energy_range = np.array([np.min(energy_grid),np.max(energy_grid)])

data      = problem.data

# fig,axes = plt.subplots(2,1)
# axes[0].plot(energy_grid,data[:,0],color="red",label="Total")
# axes[0].legend()
# axes[1].plot(energy_grid,data[:,1],color="green",label="Capture")
# axes[1].legend()
# plt.show()

result = problem.fit_Call(0, data)
print(result)
print(interaction_information["Gamma Matrix"])


# elastic_penetrability = interaction_information["Elastic Penetrability"]
# capture_penetrability = interaction_information["Capture Penetrability"]

# fudge_factor      = model_information["Fudge Factor"]






# target             = Particle(Z = int(molecular_information["Target Protons"]),
#                         A = int(molecular_information["Target Nucleons"]),
#                         mass = float(molecular_information["Target Nucleons"]),
#                         name = molecular_information["Target Name"],
#                         I = molecular_information["Target Spin"])

# spin = molecular_information["Incident Spin"] + molecular_information["Target Spin"]
# res_ladder         = pd.DataFrame({"E":       interaction_information["Resonance Levels"],
#                                 "Gn1":     interaction_information["Elastic Widths"] * 1000,
#                                 "Gg":      interaction_information["Capture Widths"] * 1000,
#                                 "J_ID":    np.ones(interaction_information["Number Levels"]),
#                                 "varyE":   np.ones(interaction_information["Number Levels"]),
#                                 "varyGg":  np.ones(interaction_information["Number Levels"]),
#                                 "varyGn1": np.ones(interaction_information["Number Levels"]),
#                                 "varyGn2": np.zeros(interaction_information["Number Levels"]),
#                                 "Jpi":     np.zeros(interaction_information["Number Levels"]) * spin,
#                                 "L":       np.zeros(interaction_information["Number Levels"])})

# part_pair          = Particle_Pair(isotope          = molecular_information["Target Name"],
#                                 resonance_ladder = res_ladder,
#                                 formalism        = "XCT",
#                                 ac               = interaction_information["Effective Radius"]/(10**-14),
#                                 energy_range     = [np.min(energy_grid),np.max(energy_grid)],
#                                 energy           = energy_grid,
#                                 target           = target,
#                                 projectile       = Neutron,
#                                 l_max            = 1)

# part_pair.add_spin_group(Jpi     = str(spin),
#                         J_ID    = 1,
#                         D       = 9,
#                         gn2_avg = interaction_information["Elastic Variance"],
#                         gn2_dof = 1,
#                         gg2_avg = interaction_information["Capture Variance"],
#                         gg2_dof = 1)






# exp_model_trans          = Experimental_Model(title        = "trans_model",
#                                         reaction     = "trans",
#                                         energy_grid  = energy_grid,
#                                         temp         = (interaction_information["Broadening Temperature"], 0))

# exp_model_trans.template = os.path.realpath("sammy_template.inp")

# trans_data = pd.DataFrame({"E":energy_grid, "exp":data[:,0], "exp_unc":np.clip(data[:,0]*fudge_factor,1e-6,None)})


# exp_model_capture          = Experimental_Model(title        = "capture_model",
#                                         reaction     = "capture",
#                                         energy_grid  = energy_grid,
#                                         temp         = (interaction_information["Broadening Temperature"], 0))

# exp_model_capture.template = os.path.realpath("sammy_template.inp")

# capture_data = pd.DataFrame({"E":energy_grid, "exp":data[:,1], "exp_unc":np.clip(data[:,1]*fudge_factor,1e-6,None)})






# sammy_path         = os.path.expanduser("~")+"/.local/Sammy/bin/sammy"
# sammy_rto          = sammy_classes.SammyRunTimeOptions(sammy_path,
#                                                     Print        = True,
#                                                     bayes        = True,
#                                                     keep_runDIR  = True,
#                                                     sammy_runDIR = 'sammy_run_dir')

# sammy_input = sammy_classes.SammyInputDataYW(
#     particle_pair = part_pair,
#     resonance_ladder = res_ladder,  

#     datasets= [trans_data,capture_data],
#     experiments = [exp_model_trans,exp_model_capture],
#     experimental_covariance= [{},{}], 
    
#     max_steps = 10,
#     iterations = 3,
#     step_threshold = 0.01,
#     autoelim_threshold = None,

#     LS = False,
#     LevMar = True,
#     LevMarV = 2,
#     LevMarVd= 5,
#     maxF= 1.5,
#     initial_parameter_uncertainty = 0.01
# )

# sammy_out = sammy_functions.run_sammy_YW(sammy_input, sammy_rto)

# true_gamma_matrix = interaction_information["Gamma Matrix"]
# fitted_gamma_matrix = np.zeros((2,3))
# fitted_gamma_matrix[:,0] = np.sqrt(np.abs(sammy_out.par_post["Gn1"].to_numpy())/(2000*elastic_penetrability))
# fitted_gamma_matrix[:,1:] = np.diag(np.sqrt(np.abs(sammy_out.par_post["Gg"].to_numpy())/(2000*capture_penetrability)))

# print(true_gamma_matrix)
# print()
# print(fitted_gamma_matrix)

# fig,axes = plt.subplots(2,1)
# axes[0].plot(energy_grid,data[:,0],color="red",label="Total")
# axes[0].plot(sammy_out.pw[0]["E"],sammy_out.pw[0]["theo_trans"],color="blue",label="SAMMY")
# axes[0].legend()
# axes[1].plot(energy_grid,data[:,1],color="green",label="Capture")
# axes[1].plot(sammy_out.pw[1]["E"],sammy_out.pw[1]["theo_xs"],color="orange",label="SAMMY")
# axes[1].legend()
# plt.show()