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
                        #    "Excited States":            [0,
                        #                                  2E2,
                        #                                  1E3,
                        #                                  1.5E3,
                        #                                  2E4,
                        #                                  2.3E4]}
                           "Excitation Model":          "Bimodal",
                           "Excitation Limits":         [2.5E4, 4E6,7.5E6],
                           "Number Excitation States":  15}

model_information       = {"Energy Grid Size":   100,
                           "Energy Grid Buffer": 2,
                           "Data Format":        "Total"}

fitting_parameters      = {"Iteration Limit":        1000,
                           "Improvement Threshold":  0.1,
                           "Initial Priority":       float(10E6),
                           "Priority Multiplier":    1.5,
                           "Priority Minimum":       float(10E-8),
                           "Priority Maximum":       float(10E16)}

selections              = {"Data Model": 2,
                           "Fit Model":  1,
                           "Fit Method": 2}


problem      = Problem(molecular_information,
                           interaction_information,
                           model_information,
                           fitting_parameters,
                           selections)

excited_states = problem.interaction_information["Excited States"]
print("Excited States:")
print(excited_states)
print()


energy      = problem.model_information["Energy Grid"]
data        = problem.data
raw_data    = problem.data_model.rawDataModel.get_Cross_Sections()[:,0]
print("True Cross Section:")
print(raw_data)
print()
print("Sampled Cross Section and Uncertainty:")
print(data)

plt.plot(energy, raw_data, label="True", c="red")
plt.errorbar(energy, data[:,0], data[:,1], fmt="_", markersize=6, label="Sampled", c="black", ecolor="grey")
plt.legend()
plt.show()

initial_guess = problem.get_Initial_Guess()

num_levels                          = interaction_information["Number Levels"]
best_fit                            = problem.fit_Call(initial_guess, data)
best_fit_matrix                     = np.zeros(problem.interaction_information["Gamma Matrix"].shape)
best_fit_matrix[:,0]                = best_fit[:num_levels]
best_fit_matrix[:,1:(num_levels+1)] = np.diag(best_fit[num_levels:])
fit_data                            = problem.fit_model.generate_Data(best_fit)

plt.plot(energy, raw_data, label="True", c="red")
plt.plot(energy, fit_data, label="Fit", c="green")
plt.errorbar(energy, data[:,0], data[:,1], fmt="_", markersize=6, label="Sampled", c="black", ecolor="grey")
plt.legend()
plt.show()

print()
print("True Gamma Matrix:")
print(problem.interaction_information["Gamma Matrix"])
print()
print("Fitted Gamma Matrix:")
print(best_fit_matrix)