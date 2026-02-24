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
                           "Excited States":            np.array([0,1000]),
                           "Gamma Matrix":              np.array([[ 0.39962476, -0.0118576 ,  0.02952645], [-0.13179563, -0.01365162, -0.02872839]]),
                           "Broadening Temperature":    300,
                           "Broaden Cross Section":     True}

model_information       = {"Energy Grid Size":   1000,
                           "Energy Grid Buffer": 2,
                           "Data Format":        "Full",
                           "Fudge Factor":       0.01}

fitting_parameters      = {"Iteration Limit":        1000,
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

result = problem.fit_Call(0, data)
print(result)
print(interaction_information["Gamma Matrix"])