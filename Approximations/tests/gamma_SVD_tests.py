import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from rmatrix import Particle
from Approximations.tools import distributions,numeric_evaluators,initial_estimates
from Approximations.models import gamma_SVD_model

separation_energy=float(7.5767E6) #ev
resonance_distance=600 #ev
resonance_avg_separation=8 #ev
gamma_variance=float(32E-3) #ev
neutron_variance=float(452.5E-3) #ev
first_excited_state=float(6.237E3) #ev
energy_grid_buffer=1 #ev

def create_leveled_model(neutron_variance,gamma_variance,resonance_avg_separation):
    model=gamma_SVD_model.Gamma_SVD(3,2)

    resonance_gap=distributions.sample_wigner_invCDF(1)*resonance_avg_separation

    neutron = Particle('n',1,0)
    gamma = Particle('g',0,0)
    target = Particle("181Ta",181,73)
    compound = Particle("181Ta", 181,73, Sn=separation_energy)
    model.set_incoming(neutron)
    model.set_outgoing(gamma)
    model.set_target(target)
    model.set_compound(compound)

    J = 3
    pi = 1  # positivie parity
    ell = 0  # only s-waves are implemented right now
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [106.78913185, 108.99600881]
    model.set_elastic_channel(J,pi,ell,radius,reduced_width_amplitudes)

    J = 3
    pi = 1  # positive parity
    ell = 0 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [2.51487027e-06, 2.49890268e-06]
    excitation = 0  # the product is left in the ground state 
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    J = 3
    pi = 1  # positive parity
    ell = 0 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [2.51487027e-06*0.8, 2.49890268e-06*0.8]
    excitation = first_excited_state  # the product is left in the 1st ex state at 0.5MeV
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    res_energies=[resonance_distance,resonance_distance+resonance_gap]
    energy_grid=np.linspace(res_energies[0]-energy_grid_buffer,res_energies[1]+energy_grid_buffer,1001)
    model.set_resonance_energies(res_energies)
    model.set_energy_grid(energy_grid)

    model.establish_spin_group()
    
    #neutron_std=neutron_gammas=np.sqrt(neutron_variance/(model.get_elastic_channel().calc_penetrability(separation_energy-first_excited_state)+
    #                                                     model.get_elastic_channel().calc_penetrability(separation_energy)))
    neutron_std=np.sqrt(neutron_variance)
    neutron_gammas=np.random.normal(0,neutron_std,2)
    gamma_1_std=np.sqrt(gamma_variance/(model.get_capture_channels()[0].calc_penetrability(separation_energy-first_excited_state)+
                                            model.get_capture_channels()[0].calc_penetrability(separation_energy)))
    gamma_2_std=np.sqrt(gamma_variance/(model.get_capture_channels()[1].calc_penetrability(separation_energy-first_excited_state)+
                                            model.get_capture_channels()[1].calc_penetrability(separation_energy)))
    gamma_gamma_1=np.random.normal(0,gamma_1_std,2)
    gamma_gamma_2=np.random.normal(0,gamma_2_std,2)
    
    gamma_matrix=np.zeros((2,3))
    gamma_matrix[:,0]=neutron_gammas
    gamma_matrix[:,1]=gamma_gamma_1
    gamma_matrix[:,2]=gamma_gamma_2
    
    model.set_gamma_matrix(gamma_matrix)
    
    return(model)

problem=create_leveled_model(neutron_variance,gamma_variance,resonance_avg_separation)
test_gamma=problem.get_gamma_matrix()
print(test_gamma)
initial_estimate=np.zeros(problem.num_resonances*3)
initial_estimate[:problem.num_resonances]=test_gamma[:,0]
u,s,vh=np.linalg.svd(test_gamma[:,1:])
initial_estimate[problem.num_resonances:problem.num_resonances*2]=u[:,0]*s[0]
initial_estimate[problem.num_resonances*2:problem.num_resonances*3]=vh[0,:]
print(initial_estimate)
error_evaluation=problem.evaluate(initial_estimate)
print(error_evaluation)
gradient=problem.derivate(initial_estimate)
print(gradient)
numeric_gradient=numeric_evaluators.numeric_gradient(initial_estimate,problem.evaluate,0.0000001)
print(numeric_gradient)