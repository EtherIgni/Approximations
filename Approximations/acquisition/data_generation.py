import numpy as np
import time
import sys, os

from Approximations.models.problem_container import Problem






def run_acquisition(batch_id,
                    number_attempts,
                    molecular_information,
                    interaction_information,
                    model_information,
                    fitting_parameters,
                    selections):
    file_name = str(selections["Data Model"]) + str(selections["Fit Model"]) + str(selections["Fit Method"])
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    try:
        f     = open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/model data.txt", "r")
        lines = f.readlines()
        if(len(lines)>0):
            last_entry = lines[-1]
            run_id     = int(last_entry[0:last_entry.find(" ")])+1
        else:
            run_id     = 1
    except:
        if(not(os.path.exists(file_path+"/run_data/"+file_name))):
            os.mkdir(file_path+"/run_data/"+file_name)
        if(not(os.path.exists(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)))):
            os.mkdir(file_path+"/run_data/"+file_name+"/batch "+str(batch_id))
        open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/model data.txt", 'w')
        open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/successful run data.txt", 'w')
        open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/failed run data.txt", 'w')
        run_id = 1



    print("Running")
    Start_time = time.time()
    for attempt in range(1, number_attempts):
        failure_text = "Passed"
        try:
            problem      = Problem(molecular_information,
                                    interaction_information,
                                    model_information,
                                    fitting_parameters,
                                    selections)
            num_levels   = interaction_information["Number Levels"]
            num_channels = len(interaction_information["Excited States"])
        except:
            mins,secs=divmod(int(time.time()-Start_time),60)
            hrs,mins=divmod(mins,60)
            print(str(run_id)+"|"+str(attempt),"No Model Generation",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
            continue



        try:
            with open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/model data.txt", "a") as text_file:
                #Gets Information about generated data model 
                resonance_energies = problem.data_model.math_model.get_resonance_energies()
                true_gamma_matrix  = problem.data_model.math_model.get_gamma_matrix()

                #Record Information
                text=str(run_id)+" "+str(attempt)+" | "
                for idx in range(1,resonance_energies.size):
                    text=text+str(resonance_energies[idx]-resonance_energies[idx-1])+" "
                text=text+"| "
                for excitation in interaction_information["Excited States"]:
                    text=text+str(problem.data_model.math_model.get_elastic_channel().calc_penetrability(interaction_information["Separation Energy"]-excitation))+" "
                text=text+"| "
                for level in range(num_levels):
                    for excitation in interaction_information["Excited States"]:
                        text=text+str(problem.data_model.math_model.get_capture_channels()[level].calc_penetrability(interaction_information["Separation Energy"]-excitation))+" "
                text=text+"| "
                for row in range(num_levels):
                    for col in range(num_channels):
                        text=text+str(true_gamma_matrix[row,col])+" "
                text=text[:-1]+"\n"
                text_file.write(text)
        except:
            mins,secs=divmod(int(time.time()-Start_time),60)
            hrs,mins=divmod(mins,60)
            print(str(run_id)+"|"+str(attempt),"No Model Logging",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
            continue



        try:
            initial_vector=problem.getInitialGuess()
            problem_data=problem.data
            best_fit_vector,iterations=problem.fit_call(initial_vector,problem_data)
            result=problem.fit_model.evaluate(best_fit_vector,problem_data)
        except Exception as e:
            try:
                the_type, the_value, the_traceback = sys.exc_info()
                with open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/failed run data.txt", "a") as text_file:
                    text_file.write(str(run_id)+" "+str(attempt)+" "+str(the_value)+"\n")
                mins,secs=divmod(int(time.time()-Start_time),60)
                hrs,mins=divmod(mins,60)
                print(str(run_id)+"|"+str(attempt),"Model Fit Failed",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
            except:
                try:
                    with open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/failed run data.txt", "a") as text_file:
                        text_file.write(str(run_id)+" "+str(attempt)+" Unresolvable Error\n")
                    mins,secs=divmod(int(time.time()-Start_time),60)
                    hrs,mins=divmod(mins,60)
                    print(str(run_id)+"|"+str(attempt),"Model Fit Failed",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
                except:
                    mins,secs=divmod(int(time.time()-Start_time),60)
                    hrs,mins=divmod(mins,60)
                    print(str(run_id)+"|"+str(attempt),"Weird Ahh Error",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
            continue



        try:
            with open(file_path+"/run_data/"+file_name+"/batch "+str(batch_id)+"/successful run data.txt", "a") as text_file:
                text=str(run_id)+" "+str(attempt)+" | "
                for val in initial_vector:
                    text=text+str(val)+" "
                text=text+"| "
                for val in best_fit_vector:
                    text=text+str(val)+" "
                text=text+"| "+str(result)+" | "+str(iterations)+"\n"
                text_file.write(text)
            mins,secs=divmod(int(time.time()-Start_time),60)
            hrs,mins=divmod(mins,60)
            print(str(run_id)+"|"+str(attempt),"Passed",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
        except:
            mins,secs=divmod(int(time.time()-Start_time),60)
            hrs,mins=divmod(mins,60)
            print(str(run_id)+"|"+str(attempt),"No Model Fit Logging",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))






molecular_information   = {"Incident Name":     "n",
                           "Incident Nucleons":  1,
                           "Incident Protons":   0,
                           "Departing Name":     "g",
                           "Departing Nucleons": 0,
                           "Departing Protons":  0,
                           "Compound Name":      "181Ta",
                           "Compound Nucleons":  181,
                           "Compound Protons":   71}

interaction_information = {"Separation Energy":         float(7.5767E6),
                           "Gamma Variance":            float(32E-3),
                           "Neutron Variance":          float(452.5E-3),
                           "Number Levels":             2,
                           "Resonance Distance":        600,
                           "Resonance Average Spacing": 8}

model_information       = {"Energy Grid Size":   2001,
                           "Energy Grid Buffer": 2}

fitting_parameters      = {"Iteration Limit":        1000,
                           "Improvement Threshold":  0.1,
                           "Initial Priority":       float(10E6),
                           "Priority Multiplier":    1.5,
                           "Priority Minimum":       float(10E-8),
                           "Priority Maximum":       float(10E16)}

selections              = {"Data Model": 1,
                           "Fit Model":  1,
                           "Fit Method": 2}
                
all_excited_states      = np.array([0,
                                    float(6.237E3),
                                    float(136.269E3),
                                    float(152.320E3),
                                    float(301.622E3),
                                    float(337.54E3)])

for id in [6]:
    
    interaction_information["Excited States"]=all_excited_states[:id+1]
    run_acquisition(batch_id=6,
                    number_attempts=10000,
                    molecular_information=molecular_information,
                    interaction_information=interaction_information,
                    model_information=model_information,
                    fitting_parameters=fitting_parameters,
                    selections=selections)