file_path            = "/run_data/breaking/"
file_name            = "random_low_100_channels.txt"
num_channels_for_run = 101
num_iterations       = 10000



import numpy     as np
from   tqdm  import tqdm
import os

from Approximations.models.problem_container import Problem

file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + file_path



molecular_information   = {"Incident Name":     "n",
                           "Incident Nucleons":  1,
                           "Incident Protons":   0,
                           "Departing Name":     "g",
                           "Departing Nucleons": 0,
                           "Departing Protons":  0,
                           "Target Name":        "181Ta",
                           "Target Nucleons":    181,
                           "Target Protons":     71}

interaction_information = {"Separation Energy":         float(7.5767E6),#float(1E6),
                           "Elastic Variance":          float(452.5E-3),
                           "Elastic Radius":            0.2,
                           "Capture Variance":          float(32E-3),
                           "Capture Radius":            0.2,
                           "Capture Ell":               0,
                           "Number Levels":             2,
                           "Resonance Distance":        600,
                           "Resonance Average Spacing": 500,
                           "Resonance Levels":          [600,
                                                         608]}

model_information       = {"Energy Grid Size":   2000,
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

interaction_information["Excited States"] = np.sort(((1-np.random.rand(int(num_channels_for_run-1)))*interaction_information["Separation Energy"]*0.1))



for idx in tqdm(range(num_iterations),
                desc="Fitting Data",
                ncols=80,
                smoothing=0):
    
    problem      = Problem(molecular_information,
                           interaction_information,
                           model_information,
                           fitting_parameters,
                           selections)
    num_levels   = interaction_information["Number Levels"]
    num_channels = num_channels_for_run
    
    true_gamma = problem.data_model.math_model.get_Gamma_Matrix()
    
    initial_vector  = problem.get_Initial_Guess()
    problem_data    = problem.data
    best_fit_vector = problem.fit_Call(initial_vector, problem_data)
    
    text_data     =                  np.array2string(best_fit_vector,separator=" ").replace('\n', '')[1:-1]+"|"
    text_data     = text_data      + np.array2string(initial_vector,separator=" ").replace('\n', '')[1:-1]+"|"
    for gamma_row in true_gamma:
        text_data = text_data      + np.array2string(gamma_row,separator=" ").replace('\n', '')[1:-1]+","
    text_data     = text_data[:-1] + "\n"
    with open(file_path+file_name, "a") as f:
        f.write(text_data)