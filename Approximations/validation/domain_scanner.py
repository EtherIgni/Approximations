import numpy as np
from models import neon_bands_3MeV_c1
from tools import initial_estimates
from copy import deepcopy
from tools import fitting

problem=neon_bands_3MeV_c1.model
true_gamma=np.copy(problem.get_gamma_matrix())
best=[problem.evaluate_multi_channel_error_gm(true_gamma),0,0,0,0]
for gamma_1_mod in np.arange(0.1,2.1,0.1):
    for gamma_2_mod in np.arange(0.1,2.1,0.1):
        for gamma_3_mod in np.arange(0.1,2.1,0.1):
            for gamma_4_mod in np.arange(0,2,0.1):
                initial_values=np.copy(true_gamma)
                initial_values[0,0]=initial_values[0,0]*gamma_1_mod
                initial_values[1,0]=initial_values[1,0]*gamma_2_mod
                initial_values[0,1]=initial_values[0,1]*gamma_3_mod
                initial_values[1,2]=initial_values[1,2]*gamma_4_mod
                iterable=np.array([[1,1,0],[1,0,1]],float)
                gradient_step=float(1000)
                best_fit_matrix=fitting.gradient_descent_half_step(initial_values,
                                                                iterable,
                                                                problem.derivative_of_SqrErr_channel_gm,
                                                                gradient_step,
                                                                problem.evaluate_multi_channel_error_gm,
                                                                float(1E-6),
                                                                [50,100],
                                                                5,
                                                                0)
                result=problem.evaluate_multi_channel_error_gm(best_fit_matrix)
                if(result<best[0]):
                    best=[result,gamma_1_mod,gamma_2_mod,gamma_3_mod,gamma_4_mod]
                print(gamma_1_mod,gamma_2_mod,gamma_3_mod,gamma_4_mod)
print(best)