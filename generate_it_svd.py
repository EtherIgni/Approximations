import numpy as np
from copy import deepcopy
from tools import fitting,distributions,initial_estimates
import matplotlib.pyplot as plt
from models import basic
from rmatrix import Particle
import numpy as np
import time
import sys, os

separation_energy=float(7.5767E6) #ev
resonance_distance=600 #ev
resonance_avg_separation=8 #ev
gamma_variance=float(32E-3) #ev
neutron_variance=float(452.5E-3) #ev
excited_states=[0, float(6.237E3),float(136.269E3),float(152.320E3),float(301.622E3),float(337.54E3)] #ev
energy_grid_buffer=20 #ev
energy_grid_size=1001

def create_leveled_model(neutron_variance,gamma_variance,resonance_avg_separation,energy_grid_size):
    number_levels=len(excited_states)
    model=basic.base_reaction()

    resonance_gaps=distributions.sample_wigner_invCDF(number_levels-1)*resonance_avg_separation

    neutron = Particle('n',1,0)
    gamma = Particle('g',0,0)
    target = Particle("181Ta",181,73)
    compound = Particle("181Ta", 181,73, Sn=separation_energy)
    model.set_incoming(neutron)
    model.set_outgoing(gamma)
    model.set_target(target)
    model.set_compound(compound)

    J = 3
    pi = 1
    ell = 0
    radius = 0.2
    reduced_width_amplitudes = np.ones(number_levels)
    model.set_elastic_channel(J,pi,ell,radius,reduced_width_amplitudes)

    for excitation in excited_states:
        J = 3
        pi = 1
        ell = 0
        radius = 0.2
        reduced_width_amplitudes = np.ones(number_levels)
        model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    res_energies=np.ones(number_levels)*resonance_distance
    res_energies[1:]+=resonance_gaps
    energy_grid=np.linspace(res_energies[0]-energy_grid_buffer,res_energies[-1]+energy_grid_buffer,energy_grid_size)
    model.set_resonance_energies(res_energies)
    model.set_energy_grid(energy_grid)

    model.establish_spin_group()
    
    neutron_std=np.sqrt(neutron_variance)
    gamma_matrix=np.zeros((number_levels,number_levels+1))
    gamma_matrix[:,0]=np.random.normal(0,neutron_std,number_levels)
    for i in range(number_levels):
        running_sum=0
        for excitation in excited_states:
            running_sum=running_sum+model.get_capture_channels()[i].calc_penetrability(separation_energy-excitation)
        gamma_std=np.sqrt(gamma_variance/running_sum)
        gamma_matrix[:,1+i]=np.random.normal(0,gamma_std,number_levels)
    
    model.set_gamma_matrix(gamma_matrix)
    
    return(model)

for attempt in range(10000):
    try:
        problem=create_leveled_model(neutron_variance,gamma_variance,resonance_avg_separation,energy_grid_size)
        num_levels=len(excited_states)
        initial_values=initial_estimates.single_value_approx(problem.get_gamma_matrix())
        # iterable=np.eye(num_levels,num_levels+1,1)
        # iterable[:,0]=1
        gradient_step=float(1000)
        best_fit_matrix,iterations=fitting.gradient_descent_half_step(initial_values,
                                                        iterable,
                                                        problem.derivative_numeric_svd,
                                                        gradient_step,
                                                        problem.evaluate_multi_channel_error_svd,
                                                        float(1E-6),
                                                        [50,100],
                                                        5,
                                                        0,
                                                        0)
        result=problem.evaluate_total_and_gamma_error_gm(problem.get_gm_from_svd(best_fit_matrix))
        resonance_sep=problem.get_resonance_energies()[1]
        true_gamma_matrix=problem.get_gamma_matrix()
        assert isinstance(resonance_sep,float), "resonance distance isn't a float"
        assert true_gamma_matrix.shape==(num_levels,num_levels+1), "true gamma matrix is wrong shape"
        assert true_gamma_matrix.dtype==float, "true gamma matrix is are not floats"
        assert best_fit_matrix.shape==(num_levels+1,num_levels+1), "fit gamma matrix is wrong shape"
        assert best_fit_matrix.dtype==float, "fit gamma matrix is are not floats"
        assert isinstance(result,float), "results value isn't a float"
        assert isinstance(iterations,int), "iteration count isn't an int"
        with open("successful runs 7 vd.txt", "a") as text_file:
            text_file.write(str(resonance_sep)+" "+
                            str(true_gamma_matrix[0,0])+" "+
                            str(true_gamma_matrix[1,0])+" "+
                            str(true_gamma_matrix[0,1])+" "+
                            str(true_gamma_matrix[1,1])+" "+
                            str(true_gamma_matrix[0,2])+" "+
                            str(true_gamma_matrix[1,2])+" "+
                            str(best_fit_matrix[0,0])+" "+
                            str(best_fit_matrix[0,1])+" "+
                            str(best_fit_matrix[0,2])+" "+
                            str(best_fit_matrix[1,0])+" "+
                            str(best_fit_matrix[1,2])+" "+
                            str(best_fit_matrix[2,2])+" "+
                            str(result)+" "+
                            str(iterations)+"\n")
        print(attempt)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        with open("failed runs.txt", "a") as text_file:
            text_file.write(str(attempt)+" "+str(exc_type)+" "+str(fname)+" "+str(exc_tb.tb_lineno)+"\n")