import numpy as np
import time
import sys, os

from Approximations.tools  import initial_estimates,fitting
from Approximations.models import reich_moore_model,gamma_SVD_model,generic_model_gen


def run_acquisition(batch_id,
                    number_attempts,
                    mode,
                    model_parameters):

    file_name=["null","rm","svd"]
    file_path=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    try:
        f = open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/model data.txt", "r")
        lines = f.readlines()
        if(len(lines)>0):
            last_entry=lines[-1]
            run_id=int(last_entry[0:last_entry.find(" ")])+1
        else:
            run_id=1
    except:
        os.mkdir(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id))
        open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/model data.txt", 'w')
        open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/successful run data.txt", 'w')
        open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/failed run data.txt", 'w')
        run_id=1

    print("Running")
    Start_time=time.time()
    for attempt in range(1,number_attempts):
        failure_text="Passed"
        try:
            problem=generic_model_gen.create_leveled_model(model_parameters["Separation Energy"],
                                                           model_parameters["Resonance Distance"],
                                                           model_parameters["Resonance Avg Separation"],
                                                           model_parameters["Gamma Variance"],
                                                           model_parameters["Neutron Variance"],
                                                           model_parameters["Excited States"],
                                                           model_parameters["Energy Grid Buffer"],
                                                           model_parameters["Energy Grid Size"],
                                                           reich_moore_model.Reich_Moore if mode==1 else gamma_SVD_model.Gamma_SVD)
            num_levels=len(model_parameters["Excited States"])
        except:
            mins,secs=divmod(int(time.time()-Start_time),60)
            hrs,mins=divmod(mins,60)
            print(str(run_id)+"|"+str(attempt),"No Model Generation",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
            continue
        try:
            with open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/model data.txt", "a") as text_file:
                resonance_energies=problem.get_resonance_energies()
                true_gamma_matrix=problem.get_gamma_matrix()
                text=str(run_id)+" "+str(attempt)+" | "
                for idx in range(1,resonance_energies.size):
                    text=text+str(resonance_energies[idx]-resonance_energies[idx-1])+" "
                text=text+"| "
                for excitation in model_parameters["Excited States"]:
                    text=text+str(problem.get_elastic_channel().calc_penetrability(model_parameters["Separation Energy"]-excitation))+" "
                text=text+"| "
                for level in range(num_levels):
                    for excitation in model_parameters["Excited States"]:
                        text=text+str(problem.get_capture_channels()[level].calc_penetrability(model_parameters["Separation Energy"]-excitation))+" "
                text=text+"| "
                for row in range(num_levels):
                    for col in range(num_levels+1):
                        text=text+str(true_gamma_matrix[row,col])+" "
                text=text[:-1]+"\n"
                text_file.write(text)
        except:
            mins,secs=divmod(int(time.time()-Start_time),60)
            hrs,mins=divmod(mins,60)
            print(str(run_id)+"|"+str(attempt),"No Model Logging",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
            continue
        try:
            if(mode==1):
                initial_vector=initial_estimates.reich_moore_guess(problem.get_gamma_matrix())
            else:
                initial_vector=initial_estimates.gamma_SVD_approx(problem.get_gamma_matrix())
            lm_multiplier=1.5
            lm_min=float(10e-8)
            lm_max=float(10e16)
            lm_constant=float(10e6)
            improvement_threshold=0.1
            lm_depth=1000
            best_fit_vector,iterations=fitting.LMA(initial_vector,
                                        problem.evaluate,
                                        problem.calc_hessian_and_gradient,
                                        lm_constant,
                                        lm_multiplier,
                                        lm_min,
                                        lm_max,
                                        improvement_threshold,
                                        lm_depth,
                                        0)
            result=problem.evaluate(best_fit_vector)
        except Exception as e:
            try:
                the_type, the_value, the_traceback = sys.exc_info()
                with open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/failed run data.txt", "a") as text_file:
                    text_file.write(str(run_id)+" "+str(attempt)+" "+str(the_value)+"\n")
                mins,secs=divmod(int(time.time()-Start_time),60)
                hrs,mins=divmod(mins,60)
                print(str(run_id)+"|"+str(attempt),"Model Fit Failed",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
            except:
                try:
                    with open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/failed run data.txt", "a") as text_file:
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
            with open(file_path+"/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/successful run data.txt", "a") as text_file:
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


model_parameters_Ta_181={"compound Name":'181Ta',
                         "N":181, ##Particles
                         "Z":71, ##Protons
                         "Separation Energy":float(7.5767E6), #ev
                         "Resonance Distance":600, #ev
                         "Resonance Avg Separation":8, #ev
                         "Gamma Variance":float(32E-3), #ev
                         "Neutron Variance":float(452.5E-3), #ev
                         "Excited States":np.array([0, float(6.237E3),float(136.269E3),float(152.320E3),float(301.622E3),float(337.54E3)]), #ev
                         "Energy Grid Buffer":20, #ev
                         "Energy Grid Size":1001}

model_parameters_Pb_208={"compound Name":'181Ta',
                         "N":208, ##Particles
                         "Z":82, ##Protons
                         "Separation Energy":float(7.36787E6), #ev
                         "Resonance Distance":2000, #ev 2E-6
                         "Resonance Avg Separation":12E3, #ev
                         "Gamma Variance":float(32E-3), #ev
                         "Neutron Variance":float(452.5E-3), #ev
                         "Excited States":np.array([0, float(2.614522E6),float(3.197711E6),float(3.475078E6),float(3.708451E6),float(3.919966E6)]), #ev
                         "Energy Grid Buffer":20, #ev
                         "Energy Grid Size":1001}

run_acquisition(batch_id=1,
                number_attempts=10000,
                mode=1,
                model_parameters=model_parameters_Pb_208)