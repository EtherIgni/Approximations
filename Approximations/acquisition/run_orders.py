import os
from tqdm import tqdm

from Approximations.tools.input_output              import input_Sheet_Translator
from Approximations.acquisition.data_fit_generation import run_iteration



run_data_path = os.getcwd()+"/Approximations/run_data"


def run_order(sheet_path):
    run_information, molecular_information, interaction_information, model_information, fitting_parameters, selections = input_Sheet_Translator(sheet_path)
    
    group_path      = run_data_path + "/" + run_information["Data Group"].lower().replace(" ","_")
    run_folder_path = group_path + "/" + run_information["Title"].lower().replace(" ","_")
    
    if(not(os.path.isdir(group_path))):
        os.mkdir(group_path)
    if(not(os.path.isdir(run_folder_path))):
        os.mkdir(run_folder_path)
    open(run_folder_path + "/model_data.txt",          'a')
    open(run_folder_path + "/successful_run_data.txt", 'a')
    open(run_folder_path + "/failed_run_data.txt",     'a')
    
    num_iterations = run_information["Number Data Sets"]
    for num in tqdm(range(num_iterations),
                desc="Fitting Data",
                ncols=80,
                smoothing=0):
        outcome, model_text, results_text = run_iteration(molecular_information,
                                                          interaction_information,
                                                          model_information,
                                                          fitting_parameters,
                                                          selections)
        
        if(outcome == "Failed w/o Model Gen"):
           with open(run_folder_path + "/failed_run_data.txt", 'a') as file:
               file.write("No Model Gen: " + results_text + "\n")
        else:
            with open(run_folder_path + "/model_data.txt", 'a') as file:
                file.write(model_text + "\n")
            if(outcome == "Failed w/ Model Gen"):
                with open(run_folder_path + "/failed_run_data.txt", 'a') as file:
                    file.write("No Fitting: " + results_text + "\n")
            else:
                with open(run_folder_path + "/successful_run_data.txt", 'a') as file:
                    file.write(results_text + "\n")