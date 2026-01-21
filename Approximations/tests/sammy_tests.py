import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from Approximations.models.problem_container import Problem

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.particle import Particle, Neutron
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.sammy_interface import sammy_classes, sammy_functions, template_creator






molecular_information   = {"Incident Name":     "n",
                           "Incident Nucleons":  1,
                           "Incident Protons":   0,
                           "Departing Name":     "g",
                           "Departing Nucleons": 0,
                           "Departing Protons":  0,
                           "Target Name":        "181Ta",
                           "Target Nucleons":    181,
                           "Target Protons":     71}

interaction_information = {"Separation Energy":         float(7.5767E6),
                           "Elastic Variance":          float(452.5E-3),
                           "Elastic Radius":            0.2,
                           "Capture Variance":          float(32E-3),
                           "Capture Radius":            0.2,
                           "Capture Ell":               0,
                           "Number Levels":             2,
                           "Resonance Distance":        200,
                           "Resonance Average Spacing": 9,
                           "Resonance Levels":          [215,
                                                         230],
                           "Excited States":            [0,
                                                         1E3],
                           "Gamma Matrix":              np.array([[0.36134995,  0.03251227,  0],
                                                                  [0.44320638,  0,           0.01699301]])}

model_information       = {"Energy Grid Size":   500,
                           "Energy Grid Buffer": 2,
                           "Data Format":        "Full"}

fitting_parameters      = {"Iteration Limit":        1000,
                           "Improvement Threshold":  0.1,
                           "Initial Priority":       float(10E6),
                           "Priority Multiplier":    1.5,
                           "Priority Minimum":       float(10E-8),
                           "Priority Maximum":       float(10E16)}

selections              = {"Data Model": 1,
                           "Fit Model":  1,
                           "Fit Method": 1}


problem      = Problem(molecular_information,
                           interaction_information,
                           model_information,
                           fitting_parameters,
                           selections)


energy_grid  = problem.model_information["Energy Grid"]
energy_range = np.array([np.min(energy_grid),np.max(energy_grid)])
data         = np.sum(problem.data[:,1:],1)






target             = Particle(Z = 71, A = 181, mass = 180.94803, name = "181Ta", I = 3.5)

res_ladder         = pd.DataFrame({"E":       [215,       230],
                                   "Gg":      [0.03251227, 0.01699301],
                                   "Gn1":     [0.36134995, 0.44320638],
                                   "J_ID":    [1,         1],
                                   "VaryE":   [1,         1],
                                   "VaryGg":  [1,         1],
                                   "VaryGn1": [1,         1],
                                   "Jpi":     [3.0,       3.0],
                                   "L":       [0,         0]})

part_pair          = Particle_Pair(isotope          = "Ta181",
                                   resonance_ladder = res_ladder,
                                   formalism        = "XCT",
                                   ac               = 0.8127,
                                   energy           = energy_grid,
                                   target           = target,
                                   projectile       = Neutron,
                                   l_max            = 1)

part_pair.add_spin_group(Jpi     = '3.0',
                         J_ID    = 1,
                         D       = 9,
                         gn2_avg = 0.45256615,
                         gn2_dof = 2,
                         gg2_avg = 0.0320,
                         gg2_dof = 2)

print(part_pair.resonance_ladder)

exp_model          = Experimental_Model(title        = "sim_model",
                                        reaction     = "capture",
                                        energy_range = energy_range,
                                        energy_grid  = energy_grid,
                                        temp         = (0, 0))

sammy_path         = os.path.expanduser("~")+"/.local/Sammy/bin/sammy"
sammy_rto          = sammy_classes.SammyRunTimeOptions(sammy_path,
                                                       Print        = True,
                                                       bayes        = False,
                                                       keep_runDIR  = True,
                                                       sammy_runDIR = 'sammy_run_dir')

template_creator.make_input_template("capture_template.inp", part_pair, exp_model, sammy_rto)
exp_model.template = os.path.realpath("capture_template.inp")






sammy_input = sammy_classes.SammyInputData(
    part_pair,
    res_ladder,
    template=exp_model.template,
    experiment=exp_model,
    energy_grid = exp_model.energy_grid
)

sammy_out = sammy_functions.run_sammy(sammy_input, sammy_rto)

sammy_gen = sammy_out.pw

plt.plot(sammy_gen["E"],sammy_gen["theo_xs"],color="Blue",label="Sammy")
# plt.plot(energy_grid,data,color="Green",label="R-Matrix")
plt.legend()
plt.show()






# from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
# from ATARI.syndat.data_classes import syndatOPT
# from ATARI.syndat.syndat_model import Syndat_Model


# syn_opt   = syndatOPT(calculate_covariance=False,
#                       explicit_covariance=False,
#                       sampleRES=False)

# syn_model = Syndat_Model(generative_experimental_model=exp_model,
#                          options=syn_opt)

# syn_model.sample(part_pair,
#                  sammyRTO=sammy_rto,
#                  num_samples=1)

# data_sample = syn_model.samples[0]
# full_data = data_sample.pw_reduced

# plt.plot(full_data["E"],full_data["true"],color="Blue",label="Sammy")
# plt.plot(energy,data,color="Green",label="R-Matrix")
# plt.errorbar(full_data["E"],full_data["exp"],full_data["exp_unc"],fmt="o",color="orange",label="Syndat")
# plt.legend()
# plt.show()






# new_ladder = pd.DataFrame({"E":       interaction_information["Resonance Levels"],
#                            "Gg":      [0.3,0.4],
#                            "Gn1":     [0.03,0.02],
#                            "J_ID":    [1,1],
#                            "VaryE":   [1,1],
#                            "VaryGg":  [1,1],
#                            "VaryGn1": [1,1]})

# print(part_pair.resonance_ladder)
# sammy_in = sammy_classes.SammyInputData(part_pair,
#                                         part_pair.resonance_ladder,
#                                         template                = os.path.realpath("capture_template.inp"),
#                                         experiment              = exp_model,
#                                         experimental_data       = full_data,
#                                         energy_grid             = energy,
#                                         experimental_covariance = {})

# sammy_in.initial_parameter_uncertainty=10

# sammy_in.experimental_data = data
# sammy_in.resonance_ladder["varyE"] = np.ones(len(part_pair.resonance_ladder))
# sammy_in.resonance_ladder["varyGg"] = np.ones(len(part_pair.resonance_ladder))
# sammy_in.resonance_ladder["varyGn1"] = np.ones(len(part_pair.resonance_ladder))

# sammy_out = sammy_functions.run_sammy(sammy_in,sammy_rto)
# print()
# print(sammy_out.par)
# print()
# print(sammy_out.est_df)





# input("Press any key to continue.")

# plt.figure()

# x = sammy_out.est_df.E
# y = sammy_out.est_df.theo
# y_err=  sammy_out.est_df.theo_unc
# plt.fill_between(x, y - y_err, y + y_err, color='r', alpha=0.5, label='Error Band')
# plt.plot(x, y, 'r', label='0K')


# x = sammy_out.est_df.E
# y = sammy_out.est_df.theo
# y_err=  sammy_out.est_df.theo_unc
# plt.fill_between(x, y - y_err, y + y_err, color='b', alpha=0.5, label='Error Band')
# plt.plot(x, y, 'b', label='300K')


# plt.ylabel("Total XS")

# plt.xlim([200,250])
# plt.legend()

# plt.xlabel('Energy (eV)')
# plt.tight_layout()
# plt.show()