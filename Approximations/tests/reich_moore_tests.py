import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from rmatrix import Particle
from Approximations.tools import numeric_evaluators,initial_estimates
from Approximations.models import reich_moore_model,generic_model_gen

separation_energy=float(7.5767E6) #ev
resonance_distance=600 #ev
resonance_avg_separation=8 #ev
gamma_variance=float(32E-3) #ev
neutron_variance=float(452.5E-3) #ev
excited_states=[0, float(6.237E3),float(136.269E3),float(152.320E3),float(301.622E3),float(337.54E3)] #ev
energy_grid_buffer=20 #ev
energy_grid_size=1001

problem=generic_model_gen.create_leveled_model(separation_energy,
                                               resonance_distance,
                                               resonance_avg_separation,
                                               gamma_variance,
                                               neutron_variance,
                                               excited_states,
                                               energy_grid_buffer,
                                               energy_grid_size,
                                               reich_moore_model.Reich_Moore)
test_gamma=problem.get_gamma_matrix()
print(test_gamma)
initial_estimate=initial_estimates.reich_moore_guess(test_gamma)
print(initial_estimate)
error_evaluation=problem.evaluate(initial_estimate)
print(error_evaluation)
print(" ")
gradient=problem.derivate(initial_estimate)
print(gradient)
print(numeric_evaluators.gradient(initial_estimate,problem.evaluate,0.000001))
print(" ")
next_val=initial_estimate-gradient*0.000000001
print(next_val)
next_evaluation=problem.evaluate(next_val)
print(next_evaluation)
print(next_evaluation-error_evaluation)