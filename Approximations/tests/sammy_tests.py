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
                           "Target Name":        "Ta181",
                           "Target Nucleons":    181,
                           "Target Protons":     71}

interaction_information = {"Separation Energy":         float(7.5767E6),
                           "Elastic Variance":          float(452.5E-3),
                           "Elastic Radius":            0.2,
                           "Capture Variance":          float(32E-3),
                           "Capture Radius":            0.2,
                           "Capture Ell":               0,
                           "Number Levels":             2,
                           "Resonance Distance":        600,
                           "Resonance Average Spacing": 500,
                           "Resonance Levels":          [215,
                                                         230],
                           "Excited States":            [0,
                                                         1E3],
                           "Gamma Matrix":              np.array([[0.36134995,  0.03251227,  0],
                                                                  [0.44320638,  0,           0.01699301]])}

#-0.0033823 -0.00900472

model_information       = {"Energy Grid Size":   100,
                           "Energy Grid Buffer": 2}

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


energy      = problem.model_information["Energy Grid"]
data        = problem.data[:,0]
uncertainty = np.zeros(data.size)





# full_data = pd.DataFrame({"E":energy, "exp":data, "exp_unc":uncertainty})
# print(full_data)
# print(problem.get_Initial_Guess())

# target = Particle(Z = molecular_information["Target Protons"], A = molecular_information["Target Nucleons"], mass = 180.94803, name = molecular_information["Target Name"], I = 3.5)
# part_pair = Particle_Pair(isotope          = "Ta181",
#                           resonance_ladder = pd.DataFrame({"E":       interaction_information["Resonance Levels"],
#                                                            "Gg":      [0.36134995,0.44320638],
#                                                            "Gn1":     [0.03251227,0.01699301],
#                                                            "J_ID":    [1,1],
#                                                            "VaryE":   [1,1],
#                                                            "VaryGg":  [1,1],
#                                                            "VaryGn1": [1,1]}),
#                           formalism        = "XCT",
#                           energy_grid      = energy,
#                           ac               = 0.8127,
#                           target           = target,
#                           projectile       = Neutron,
#                           l_max            = 1)
# part_pair.add_spin_group(Jpi     = '3.0',
#                          J_ID    = 1,
#                          D       = interaction_information["Resonance Average Spacing"],
#                          gn2_avg = float(452.5E-3),
#                          gn2_dof = 1,
#                          gg2_avg = float(32E-3),
#                          gg2_dof = 2)

# exp_model = Experimental_Model(title        = "sim_model",
#                                reaction     = "capture",
#                                energy_range = [np.min(energy), np.max(energy)],
#                                energy_grid  = energy)

# sammy_path = "/home/aaron/Depo/SAMMY/sammy/build/bin/sammy"
# sammy_rto  = sammy_classes.SammyRunTimeOptions(sammy_path,
#                                                Print        = True,
#                                                bayes        = True,
#                                                keep_runDIR  = True,
#                                                sammy_runDIR = 'sammy_run_dir')

# full_data = pd.DataFrame({"E":energy, "exp":data, "exp_unc":uncertainty})
# print(full_data)
# print(problem.get_Initial_Guess())





target = Particle(Z = molecular_information["Target Protons"], A = molecular_information["Target Nucleons"], mass = 180.94803, name = molecular_information["Target Name"], I = 3.5)
part_pair = Particle_Pair(isotope          = "Ta181",
                          resonance_ladder = pd.DataFrame({"E":       interaction_information["Resonance Levels"],
                                                           "Gg":      [0.36134995,0.44320638],
                                                           "Gn1":     [0.03251227,0.01699301],
                                                           "J_ID":    [1,1],
                                                           "VaryE":   [1,1],
                                                           "VaryGg":  [1,1],
                                                           "VaryGn1": [1,1]}),
                          formalism        = "XCT",
                          energy_grid      = energy,
                          ac               = 0.8127,
                          target           = target,
                          projectile       = Neutron,
                          l_max            = 1)
part_pair.add_spin_group(Jpi='3.0',
                         J_ID=1,
                         D=9.0030,
                         gn2_avg=452.56615, #46.5,
                         gn2_dof=1,
                         gg2_avg=32.0,
                         gg2_dof=2)

exp_model = Experimental_Model(title        = "sim_model",
                               reaction     = "transmission",
                               energy_range = [200,250])

sammy_path = "/home/aaron/Depo/SAMMY/sammy/build/bin/sammy"
sammy_rto  = sammy_classes.SammyRunTimeOptions(sammy_path,
                                               Print        = True,
                                               bayes        = True,
                                               keep_runDIR  = True,
                                               sammy_runDIR = 'sammy_run_dir')

template_creator.make_input_template("capture_template.inp", part_pair, exp_model, sammy_rto)
exp_model.template = os.path.realpath("capture_template.inp")





### Generate syndat from measurement models
from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.syndat.data_classes import syndatOPT
from ATARI.syndat.syndat_model import Syndat_Model


syn_opt   = syndatOPT(calculate_covariance=False,
                     explicit_covariance=False)

syn_model = Syndat_Model(
              generative_experimental_model=exp_model,
              options=syn_opt)

syn_model.sample(part_pair,
                 sammyRTO=sammy_rto,
                 num_samples=1)

data_sample = syn_model.samples[0]
full_data = data_sample.pw_reduced

plt.plot(full_data["E"],full_data["true"])
plt.errorbar(full_data["E"],full_data["exp"],full_data["exp_unc"],fmt="o")
plt.show()





new_ladder = pd.DataFrame({"E":       interaction_information["Resonance Levels"],
                           "Gg":      [0.3,0.4],
                           "Gn1":     [0.03,0.02],
                           "J_ID":    [1,1],
                           "VaryE":   [1,1],
                           "VaryGg":  [1,1],
                           "VaryGn1": [1,1]})

print(part_pair.resonance_ladder)
sammy_in = sammy_classes.SammyInputData(part_pair,
                                        part_pair.resonance_ladder,
                                        template                = os.path.realpath("capture_template.inp"),
                                        experiment              = exp_model,
                                        experimental_data       = full_data,
                                        energy_grid             = energy,
                                        experimental_covariance = {})

sammy_in.initial_parameter_uncertainty=10

sammy_in.experimental_data = data
sammy_in.resonance_ladder["varyE"] = np.ones(len(part_pair.resonance_ladder))
sammy_in.resonance_ladder["varyGg"] = np.ones(len(part_pair.resonance_ladder))
sammy_in.resonance_ladder["varyGn1"] = np.ones(len(part_pair.resonance_ladder))

sammy_out = sammy_functions.run_sammy(sammy_in,sammy_rto)
print()
print(sammy_out.par)
print()
print(sammy_out.est_df)





input("Press any key to continue.")

plt.figure()

x = sammy_out.est_df.E
y = sammy_out.est_df.theo
y_err=  sammy_out.est_df.theo_unc
plt.fill_between(x, y - y_err, y + y_err, color='r', alpha=0.5, label='Error Band')
plt.plot(x, y, 'r', label='0K')


x = sammy_out.est_df.E
y = sammy_out.est_df.theo
y_err=  sammy_out.est_df.theo_unc
plt.fill_between(x, y - y_err, y + y_err, color='b', alpha=0.5, label='Error Band')
plt.plot(x, y, 'b', label='300K')


plt.ylabel("Total XS")

plt.xlim([200,250])
plt.legend()

plt.xlabel('Energy (eV)')
plt.tight_layout()
plt.show()