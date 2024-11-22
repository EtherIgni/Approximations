batch_id=3
number_attempts=10000
mode=2 #1:Reich-Moore, 2:gamma SVD

separation_energy=float(7.5767E6) #ev
resonance_distance=600 #ev
resonance_avg_separation=8 #ev
gamma_variance=float(32E-3) #ev
neutron_variance=float(452.5E-3) #ev
excited_states=[0, float(6.237E3)]#,float(136.269E3),float(152.320E3),float(301.622E3),float(337.54E3)] #ev
energy_grid_buffer=20 #ev
energy_grid_size=1001







import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import sys, os

from Approximations.tools  import initial_estimates,fitting
from Approximations.models import reich_moore_model,gamma_SVD_model,generic_model_gen


file_name=["null","rm","svd"]
try:
    f = open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/model data.txt", "r")
    lines = f.readlines()
    last_entry=lines[-1]
    run_id=int(last_entry[0:last_entry.find(" ")])+1
except:
    os.mkdir("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id))
    open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/model data.txt", 'w')
    open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/successful run data.txt", 'w')
    open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/failed run data.txt", 'w')
    run_id=1

print("Running")
Start_time=time.time()
for attempt in range(1,number_attempts):
    failure_text="Passed"
    try:
        problem=generic_model_gen.create_leveled_model(separation_energy,
                                                       resonance_distance,
                                                       resonance_avg_separation,
                                                       gamma_variance,
                                                       neutron_variance,
                                                       excited_states,
                                                       energy_grid_buffer,
                                                       energy_grid_size,
                                                       reich_moore_model.Reich_Moore if mode==1 else gamma_SVD_model.Gamma_SVD)
        num_levels=len(excited_states)
    except:
        mins,secs=divmod(int(time.time()-Start_time),60)
        hrs,mins=divmod(mins,60)
        print(str(run_id)+"|"+str(attempt),"No Model Generation",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
        continue
    try:
        with open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/model data.txt", "a") as text_file:
            resonance_energies=problem.get_resonance_energies()
            true_gamma_matrix=problem.get_gamma_matrix()
            text=str(run_id)+" "+str(attempt)+" | "
            for idx in range(1,resonance_energies.size):
                text=text+str(resonance_energies[idx]-resonance_energies[idx-1])+" "
            text=text+"| "
            for excitation in excited_states:
                text=text+str(problem.get_elastic_channel().calc_penetrability(separation_energy-excitation))+" "
            text=text+"| "
            for level in range(num_levels):
                for excitation in excited_states:
                    text=text+str(problem.get_capture_channels()[level].calc_penetrability(separation_energy-excitation))+" "
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
            initial_values=initial_estimates.reich_moore_guess(problem.get_gamma_matrix())
        else:
            initial_values=initial_estimates.gamma_SVD_approx(problem.get_gamma_matrix())
        iterable=np.ones(initial_values.shape)
        gradient_step=float(1000)
        best_fit_matrix,iterations=fitting.gradient_descent_half_step(initial_values,
                                                                      iterable,
                                                                      problem.derivate,
                                                                      gradient_step,
                                                                      problem.evaluate,
                                                                      float(1E-6),
                                                                      [100,100],
                                                                      5,
                                                                      0,
                                                                      0)
        result=problem.evaluate(best_fit_matrix)
    except Exception as e:
        try:
            the_type, the_value, the_traceback = sys.exc_info()
            with open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/failed run data.txt", "a") as text_file:
                text_file.write(str(run_id)+" "+str(attempt)+" "+str(the_value)+"\n")
            mins,secs=divmod(int(time.time()-Start_time),60)
            hrs,mins=divmod(mins,60)
            print(str(run_id)+"|"+str(attempt),"Model Fit Failed",'{:d}:{:02d}:{:02d}'.format(hrs,mins,secs))
        except:
            try:
                with open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/failed run data.txt", "a") as text_file:
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
        with open("Approximations/run data/"+file_name[mode]+"/batch "+str(batch_id)+"/successful run data.txt", "a") as text_file:
            text=str(run_id)+" "+str(attempt)+" | "
            for val in initial_values:
                text=text+str(val)+" "
            text=text+"| "
            for val in best_fit_matrix:
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
        