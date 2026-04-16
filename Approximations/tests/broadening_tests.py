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






def calc_diff_at_temp(temperature):
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
                            #    "Excitation Model":          "Flat",
                            #    "Excitation Limits":         np.array([1E3]),
                            #    "Number Excitation States":  6,
                            "Gamma Matrix":              np.array([[0.50140825, 0.03331835, -0.0492754, -0.03185534, -0.03076693, -0.07564733, 0.01034401],
                                                                    [-0.7632555, -0.05900402, -0.08273421, 0.01095067, -0.01600579, 0.00084774, -0.04459883]]),
                            "Broadening Temperature":    temperature,
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

    interaction_information["Excited States"] = np.array([0, 53.07908286, 167.47903885, 346.92028851, 350.67468498, 880.06569417])

    problem_1 = Problem(molecular_information,
                        interaction_information,
                        model_information,
                        fitting_parameters,
                        selections)

    interaction_information["Excited States"] = np.array([0, 2.24578566E2, 7.26019895E2, 6.14463868E6, 6.42822818E6, 6.87371751E6])

    problem_2 = Problem(molecular_information,
                        interaction_information,
                        model_information,
                        fitting_parameters,
                        selections)



    energy_grid = problem_1.model_information["Energy Grid"]

    differential_data = np.power(problem_1.data-problem_2.data, 2)
    return(differential_data)


temperature_spread = np.linspace(0,300,101)[1:]
differential_grid = np.zeros((100,600,2))

for i in range(100):
    differential_grid[i] = calc_diff_at_temp(temperature_spread[i])[::10]

np.save(os.getcwd()+"/Approximations/tests/saved_differential_grid.npy",differential_grid)