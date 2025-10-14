import numpy as np

from Approximations.models.problem_container import Problem



def matrix_to_string(matrix):
    return(np.array2string(matrix,
                           separator      = ', ',
                           suppress_small = False)).replace('\n', '')

def run_iteration(molecular_information,
                  interaction_information,
                  model_information,
                  fitting_parameters,
                  selections):
    
    # Sets up model and creates true data
    try:
        problem = Problem(molecular_information,
                        interaction_information,
                        model_information,
                        fitting_parameters,
                        selections)
        res_lvls_txt = str(problem.interaction_information["Resonance Levels"])
        true_gm_text = matrix_to_string(problem.true_gamma_matrix)
        model_text   = res_lvls_txt + " | " + true_gm_text
    except Exception as exc:
        return("Failed w/o Model Gen", "", str(exc))
    
    # Fits data
    try:
        initial_vector  = problem.get_Initial_Guess()
        problem_data    = problem.data
        best_fit_vector = problem.fit_Call(initial_vector, problem_data)
        
        initial_text = matrix_to_string(initial_vector)
        fit_text     = matrix_to_string(best_fit_vector)
        results_text = initial_text + " | " + fit_text
        
        return("Finished w/o Failure", model_text, results_text)
    except Exception as exc:
        return("Failed w/ Model Gen", model_text, str(exc))