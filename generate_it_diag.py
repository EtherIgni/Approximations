import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import sys, os

from Approximations.tools  import initial_estimates,fitting
from Approximations.models import reich_moore_model,generic_model_gen

run_number=1
separation_energy=float(7.5767E6) #ev
resonance_distance=600 #ev
resonance_avg_separation=8 #ev
gamma_variance=float(32E-3) #ev
neutron_variance=float(452.5E-3) #ev
excited_states=[0, float(6.237E3),float(136.269E3),float(152.320E3),float(301.622E3),float(337.54E3)] #ev
energy_grid_buffer=20 #ev
energy_grid_size=1001

for attempt in range(1,10000):
    try:
        problem=generic_model_gen.create_leveled_model(separation_energy,
                                                       resonance_distance,
                                                       resonance_avg_separation,
                                                       gamma_variance,
                                                       neutron_variance,
                                                       excited_states,
                                                       energy_grid_buffer,
                                                       energy_grid_size,
                                                       reich_moore_model.Reich_Moore)
        num_levels=len(excited_states)
        with open("Run Data/Diag/Run Data 1.txt", "a") as text_file:
            resonance_energies=problem.get_resonance_energies()[1]
            true_gamma_matrix=problem.get_gamma_matrix()
            text=str(attempt)+" "
            for idx in range(1,resonance_energies.size):
                text=text+str(resonance_energies[idx]-resonance_energies[idx-1])+" "
            for excitation in excited_states:
                text=text+str(problem.get_elastic_channel().calc_penetrability(separation_energy-excitation))
            for level in range(num_levels):
                for excitation in excited_states:
                    text=text+str(problem.get_capture_channels()[level].calc_penetrability(separation_energy-excitation))
            for row in range(num_levels):
                for col in range(num_levels+1):
                    text=text+str(true_gamma_matrix[row,col])+" "
            text=text+"\n"
            text_file.write(text)
        try:
            initial_values=initial_estimates.reich_moore_guess(problem.get_gamma_matrix())
            iterable=np.ones(initial_values.shape)
            gradient_step=float(1000)
            best_fit_matrix,iterations=fitting.gradient_descent_half_step(initial_values,
                                                                          iterable,
                                                                          problem.derivative_total_and_gamma_error_gm,
                                                                          gradient_step,
                                                                          problem.evaluate_total_and_gamma_error_gm,
                                                                          float(1E-6),
                                                                          [100,100],
                                                                          5,
                                                                          0,
                                                                          0)
            result=problem.evaluate_total_and_gamma_error_gm(best_fit_matrix)
            with open("Run Data/Diag/Successful Runs "+str(run_number)+".txt", "a") as text_file:
                text=str(attempt)+" "
                row_list,col_list=iterable.nonzero()
                for i in range(row_list.size):
                    text=text+str(best_fit_matrix[row_list[i],col_list[i]])+" "
                text=text+str(result)+" "+str(iterations)+"\n"
                text_file.write(text)
            print(attempt,"Ran")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            with open("Run Data/Diag/Failed Runs "+str(run_number)+".txt", "a") as text_file:
                text_file.write(str(attempt)+" "+str(exc_type)+" "+str(fname)+" "+str(exc_tb.tb_lineno)+"\n")
    except:
        print(attempt,"No Run")