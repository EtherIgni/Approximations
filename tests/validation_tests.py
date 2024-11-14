import numpy as np
from copy import deepcopy
from tools import fitting,distributions,initial_estimates
import matplotlib.pyplot as plt
from models import basic
from rmatrix import Particle
import numpy as np
import time

separation_energy=float(7.5767E6) #ev
resonance_distance=600 #ev
resonance_avg_separation=8 #ev
gamma_variance=float(32E-3) #ev
neutron_variance=float(452.5E-3) #ev
first_excited_state=float(6.237E3) #ev
energy_grid_buffer=1 #ev

def create_leveled_model(neutron_variance,gamma_variance,resonance_avg_separation):
    model=basic.base_reaction()

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
    
    L_matrix=model.get_L_matrix()
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
test_gamma[0,0]=1.001
iterable_mapping=np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]],int)



print(problem.get_gamma_matrix())

#test_gamma=np.ones(problem.get_gamma_matrix().shape)
print(problem.evaluate_total_and_gamma_error_gm(test_gamma))
print("")
for iteration in range(300):
    gradient,hessian=problem.calc_hessian_and_gradient(test_gamma,iterable_mapping)
    #print(gradient,hessian)
    delta_grad=-np.linalg.inv(hessian)@gradient
    delta_mat=np.zeros(test_gamma.shape)
    for i in range(delta_grad.size):
        delta_mat[iterable_mapping[i,0],iterable_mapping[i,1]]=delta_grad[i]
    test_gamma=test_gamma+delta_mat
    print(np.sum(np.power(gradient,2)))
print("")
print(test_gamma)
print(problem.evaluate_total_and_gamma_error_gm(test_gamma))
print(np.sum(np.power(gradient,2)))









# problem=neon_modified.model
# problem_name="neon"

# test_gamma=np.array([[np.random.normal(problem.get_gamma_matrix()[0,0],10),np.random.normal(problem.get_gamma_matrix()[1,1],float(1E-5)),problem.get_gamma_matrix()[1,2]],
#                           [np.random.normal(problem.get_gamma_matrix()[1,0],10),problem.get_gamma_matrix()[1,1],np.random.normal(problem.get_gamma_matrix()[1,2],float(1E-7))]],float)

# print(problem.evaluate_total_and_gamma_error_gm(problem.get_gamma_matrix()))
# print(problem.evaluate_total_and_gamma_error_gm(test_gamma))
# numeric=problem.derivative_numeric_gm(test_gamma,problem.evaluate_total_and_gamma_error_gm,0.00001)
# analytic=problem.derivative_total_and_gamma_error_gm(test_gamma)
# print(numeric)
# print(analytic)
# print(numeric-analytic)
# print(np.max(numeric-analytic))


# channel_heights=np.empty(len(sg.channels))
# for idx in range(channel_heights.size):
#     channel_heights[idx]=np.mean(sg.channels[idx].cross_section)
# channel_heights=channel_heights/channel_heights[0]
# print(channel_heigjhts)
# channel_heights[0]=0.001

#Fitting

# def cross_chan_error(gamma_matrix):
#     new_sg = deepcopy(sg)
#     new_sg.update_gamma_matrix(gamma_matrix)
#     total_cross_section=np.zeros(new_sg.energy_grid.shape)
#     for idc,channel in enumerate(new_sg.channels):
#         total_cross_section+=(sg.channels[idc].cross_section-channel.cross_section)
#     return(total_cross_section)

# def derivative_cross(gamma_matrix,gamma_der):
#     new_sg=deepcopy(sg)
#     new_sg.update_gamma_matrix(gamma_matrix)
#     ana_A_der=-(gamma_der@new_sg.L_matrix@new_sg.gamma_matrix.T+new_sg.gamma_matrix@new_sg.L_matrix@gamma_der.T)
#     ana_A_inv_der=-(new_sg.A_matrix@ana_A_der@new_sg.A_matrix)
#     ana_W_der=2j*new_sg.P_half@(gamma_der.T@new_sg.A_matrix@new_sg.gamma_matrix+
#                             new_sg.gamma_matrix.T@ana_A_inv_der@new_sg.gamma_matrix+
#                             new_sg.gamma_matrix.T@new_sg.A_matrix@gamma_der)@new_sg.P_half
#     ana_U_der=new_sg.omega_matrix@ana_W_der@new_sg.omega_matrix
#     ana_chan_0_der=10**24 * np.pi/new_sg.k_sq*(-2*ana_U_der[:,0,0].real+np.conjugate(ana_U_der[:,0,0])*new_sg.U_matrix[:,0,0]+np.conjugate(new_sg.U_matrix[:,0,0])*ana_U_der[:,0,0]).real
#     ana_chan_1_der=10**24 * np.pi/new_sg.k_sq *(np.conjugate(ana_U_der[:,0,1])*new_sg.U_matrix[:,0,1]+np.conjugate(new_sg.U_matrix[:,0,1])*ana_U_der[:,0,1]).real
#     ana_chan_2_der=10**24 * np.pi/new_sg.k_sq *(np.conjugate(ana_U_der[:,0,2])*new_sg.U_matrix[:,0,2]+np.conjugate(new_sg.U_matrix[:,0,2])*ana_U_der[:,0,2]).real
#     ana_der=ana_chan_0_der+ana_chan_1_der+ana_chan_2_der
#     return(ana_der)

# def calc_jacobian(energy_length,gamma_matrix):
#     num_cols=gamma_matrix.shape[1]
#     jac=np.empty((energy_length,gamma_matrix.size))
#     for idr,row in enumerate(gamma_matrix):
#         for idc,element in enumerate(row):
#             gamma_der=np.zeros(gamma_matrix.shape)
#             gamma_der[idr,idc]=1
#             jac[:,idr*num_cols+idc]=derivative_cross(gamma_matrix,gamma_der)
#     return(jac)
            

# gm=sg.gamma_matrix
# gm[0,1]+=0
# xs=cross_chan_error(gm)
# der=np.zeros(sg.gamma_matrix.shape)
# der[0,0]=1
# der_xs=derivative_cross(sg.gamma_matrix,der)

# print(xs)
# print(der_xs)
# print(sg.energy_grid.shape)
# a=np.zeros((2,6))
# print(a.size)
# J=calc_jacobian(sg.energy_grid.size,gm)
# print(J.shape)
# print(np.linalg.inv(J.T@J)@J.T@cross_chan_error(gm))

# fidelity=1000
# element=[1,0]
# x=np.linspace(0,sg.gamma_matrix[element[0],element[1]]*4,fidelity)
# y=np.empty(fidelity)
# for idx,val in enumerate(x):
#     matrix=np.copy(sg.gamma_matrix)
#     matrix[element[0],element[1]]=val
#     if((idx==3)or(idx==103)):
#         print(matrix)
#     y[idx]=evaluator(matrix)

# plt.plot(x,y)
# plt.show()

# new_sg=deepcopy(sg)
# new_matrix=np.copy(sg.gamma_matrix)
# new_matrix[0,0]-=20
# new_sg.update_gamma_matrix(new_matrix)

# element=[0,0]

# gamma_der=np.zeros((2,3))
# gamma_der[element[0],element[1]]=1
# ana_A_der=-(gamma_der@new_sg.L_matrix@new_sg.gamma_matrix.T+new_sg.gamma_matrix@new_sg.L_matrix@gamma_der.T)
# ana_A_inv_der=-(new_sg.A_matrix@ana_A_der@new_sg.A_matrix)
# ana_W_der=2j*new_sg.P_half@(gamma_der.T@new_sg.A_matrix@new_sg.gamma_matrix+
#                         new_sg.gamma_matrix.T@ana_A_inv_der@new_sg.gamma_matrix+
#                         new_sg.gamma_matrix.T@new_sg.A_matrix@gamma_der)@new_sg.P_half
# ana_U_der=new_sg.omega_matrix@ana_W_der@new_sg.omega_matrix
# # ana_real_der=np.real(ana_U_der)
# # ana_cross_der=-10**24*2*np.pi/new_sg.k_sq*ana_real_der[:,0,0]
# # ana_SE_der=np.sum(-2*(sg.total_cross_section-new_sg.total_cross_section)*ana_cross_der)
# ana_chan_0_der=10**24 * np.pi/new_sg.k_sq*(-2*ana_U_der[:,0,0].real+np.conjugate(ana_U_der[:,0,0])*new_sg.U_matrix[:,0,0]+np.conjugate(new_sg.U_matrix[:,0,0])*ana_U_der[:,0,0]).real
# ana_chan_1_der=10**24 * np.pi/new_sg.k_sq *(np.conjugate(ana_U_der[:,0,1])*new_sg.U_matrix[:,0,1]+np.conjugate(new_sg.U_matrix[:,0,1])*ana_U_der[:,0,1]).real
# ana_chan_2_der=10**24 * np.pi/new_sg.k_sq *(np.conjugate(ana_U_der[:,0,2])*new_sg.U_matrix[:,0,2]+np.conjugate(new_sg.U_matrix[:,0,2])*ana_U_der[:,0,2]).real
# ana_SE_der=np.sum(-2*(sg.channels[0].cross_section-new_sg.channels[0].cross_section)*ana_chan_0_der-
#                   2*(sg.channels[1].cross_section-new_sg.channels[1].cross_section)*ana_chan_1_der-
#                   2*(sg.channels[2].cross_section-new_sg.channels[2].cross_section)*ana_chan_2_der)


# numeric_step=float(1E-10)
# new_matrix=np.copy(sg.gamma_matrix)
# new_matrix[element[0],element[1]]-=20
# new_matrix[element[0],element[1]]+=numeric_step
# new_sg.update_gamma_matrix(new_matrix)
# U_forward=new_sg.U_matrix
# A_forward=new_sg.A_inv
# A_inv_forward=new_sg.A_matrix
# W_forward=new_sg.W_matrix
# real_forward=np.real(U_forward)
# cross_forward=new_sg.total_cross_section
# SE_forward=evaluator(new_matrix)
# chan_forward=new_sg.channels[0].cross_section

# new_matrix=np.copy(sg.gamma_matrix)
# new_matrix[element[0],element[1]]-=20
# new_matrix[element[0],element[1]]-=numeric_step
# new_sg.update_gamma_matrix(new_matrix)
# U_backwards=new_sg.U_matrix
# A_backwards=new_sg.A_inv
# A_inv_backwards=new_sg.A_matrix
# W_backwards=new_sg.W_matrix
# real_backwards=np.real(U_backwards)
# cross_backwards=new_sg.total_cross_section
# SE_backwards=evaluator(new_matrix)
# chan_backwards=new_sg.channels[0].cross_section

# num_U_der=(U_forward-U_backwards)/(numeric_step*2)
# num_A_der=(A_forward-A_backwards)/(numeric_step*2)
# num_A_inv_der=(A_inv_forward-A_inv_backwards)/(numeric_step*2)
# num_W_der=(W_forward-W_backwards)/(numeric_step*2)
# num_real_der=(real_forward-real_backwards)/(numeric_step*2)
# num_cross_der=(cross_forward-cross_backwards)/(numeric_step*2)
# num_SE_der=(SE_forward-SE_backwards)/(numeric_step*2)
# num_chan_der=(chan_forward-chan_backwards)/(numeric_step*2)

# print(num_chan_der[50])
# print("")
# print(ana_chan_0_der[50])
# print("")
# print(num_SE_der)
# print("")
# print(ana_SE_der)

#coordinate Descent
# initial_values=np.array([[2,2,2],[2,2,2]],float)
# high_step=100
# low_step=0.5
# steps=np.array([[high_step,low_step,0],[high_step,0,low_step]],float)
# best_fit_matrix=fitting.coordinate_descent(initial_values,
#                                          steps,
#                                          evaluator,
#                                          5000,
#                                          False)

#Gradient Descent Numerical
# initial_values=np.array([[100,0.00005,0.00005],[100,0.00005,0.00005]],float)
# iterable=np.array([[1,1,1],[1,1,1]],float)
# finite_step=float(1E-8)
# gradient_step=float(1)
# best_fit_matrix=fitting.gradient_descent_numeric(initial_values,
#                                          iterable,
#                                          finite_step,
#                                          gradient_step,
#                                          evaluator,
#                                          float(1E-50),
#                                          500,
#                                          True)

#Gradient Descent Type 2
# initial_values=np.array([[1,1,1],[1,1,1]],float)
# # initial_values=diag_best_approx
# iterable=np.array([[1,1,1],[1,1,1]],float)
# gradient_step=float(100)
# best_fit_matrix=fitting.gradient_descent_multi_sampled(initial_values,
#                                                   iterable,
#                                                   derivative_SE,
#                                                   gradient_step,
#                                                   evaluator,
#                                                   float(1E-10),
#                                                   3,
#                                                   15,
#                                                   500,
#                                                   1)

# Gradient Descent Type 1
#initial_values=np.array([[1,1,1],[1,1,1]],float)
#initial_values=np.array([[np.random.normal(problem.get_gamma_matrix()[0,0],10),np.random.normal(problem.get_gamma_matrix()[1,1],float(100)),np.random.normal(problem.get_gamma_matrix()[1,2],float(1000))],
#                          [np.random.normal(problem.get_gamma_matrix()[1,0],10),np.random.normal(problem.get_gamma_matrix()[1,1],float(100)),np.random.normal(problem.get_gamma_matrix()[1,2],float(1000))]],float)



# final_gradient=problem.derivative_real_channel_error_gm(test_gamma)
# numeric_gradient=problem.derivative_numeric(test_gamma,problem.evaluate_real_channel_error_gm,0.0001)
# print(final_gradient)
# print(numeric_gradient)

# ive=initial_estimates.single_value_approx(problem.get_gamma_matrix())
# print(problem.evaluate_multi_channel_error_svd(ive))
# print(ive)
# print(problem.get_gm_from_svd(ive))
# print(problem.get_gamma_matrix())
# print(np.linalg.svd(problem.get_gamma_matrix()))

# iterable=np.array([[1,1,1],[1,0,1],[1,0,0]],float)
# gradient_step=float(1000)
# best_svd=fitting.gradient_descent_half_step(ive,
#                                                     iterable,
#                                                     problem.derivative_numeric_svd,
#                                                     gradient_step,
#                                                     problem.evaluate_multi_channel_error_svd,
#                                                     float(1E-6),
#                                                     [50,100],
#                                                     5,
#                                                     0,
#                                                     0)

# print(best_svd)
# print(problem.evaluate_multi_channel_error_svd(best_svd))