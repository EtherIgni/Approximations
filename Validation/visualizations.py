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

data=np.genfromtxt("successful runs 4.txt",float,delimiter=" ")
val=-1
true_gammas=data[val,2:8]
gamma_vals=data[val,8:12]
# print(gamma_vals)
test_matrix=np.array([[gamma_vals[0],gamma_vals[2],0],[gamma_vals[1],0,gamma_vals[3]]])
true=np.array([[true_gammas[0],true_gammas[2],true_gammas[4]],[true_gammas[1],true_gammas[3],true_gammas[5]]])
problem=create_leveled_model(neutron_variance,gamma_variance,resonance_avg_separation)
# print(problem.get_gamma_matrix())

gam=problem.get_gamma_matrix()
L=problem.get_L_matrix()/(1j)
print(gam)
print(gam@L@gam.T)

# print(test_matrix)
# print(true)
# print(data[val,-1])

true_sg = deepcopy(problem.get_spin_group())
true_sg.update_gamma_matrix(true)
fitted_sg = deepcopy(problem.get_spin_group())
fitted_sg.update_gamma_matrix(test_matrix)

fig,ax=plt.subplots(2,3)
fig.set_figheight(10)
fig.set_figwidth(20)

ax[0,0].plot(problem.get_energy_grid(), true_sg.channels[0].cross_section, c="b")
ax[0,0].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section, c="r",linestyle="dashed")
ax[0,0].set_ylabel("Cross Section [b]")
ax[0,0].set_xlabel("Incident Neutron Energy [eV]")
ax[0,0].set_title("Channel 0")

ax[0,1].plot(problem.get_energy_grid(), true_sg.channels[1].cross_section, c="b")
ax[0,1].plot(problem.get_energy_grid(), fitted_sg.channels[1].cross_section, c="r",linestyle="dashed")
ax[0,1].set_ylabel("Cross Section [b]")
ax[0,1].set_xlabel("Incident Neutron Energy [eV]")
ax[0,1].set_title("Channel 1")

ax[0,2].plot(problem.get_energy_grid(), true_sg.channels[2].cross_section, c="b")
ax[0,2].plot(problem.get_energy_grid(), fitted_sg.channels[2].cross_section, c="r",linestyle="dashed")
ax[0,2].set_ylabel("Cross Section [b]")
ax[0,2].set_xlabel("Incident Neutron Energy [eV]")
ax[0,2].set_title("Channel 2")

ax[1,0].plot(problem.get_energy_grid(), true_sg.channels[0].cross_section+true_sg.channels[1].cross_section+true_sg.channels[2].cross_section, c="b")
ax[1,0].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section+fitted_sg.channels[1].cross_section+fitted_sg.channels[2].cross_section, c="r",linestyle="dashed")
ax[1,0].set_ylabel("Cross Section [b]")
ax[1,0].set_xlabel("Incident Neutron Energy [eV]")
ax[1,0].set_title("Total")

ax[1,1].plot(problem.get_energy_grid(), true_sg.channels[1].cross_section+true_sg.channels[2].cross_section, c="b")
ax[1,1].plot(problem.get_energy_grid(), fitted_sg.channels[1].cross_section+fitted_sg.channels[2].cross_section, c="r",linestyle="dashed")
ax[1,1].set_ylabel("Cross Section [b]")
ax[1,1].set_xlabel("Incident Neutron Energy [eV]")
ax[1,1].set_title("Gamma")

plt.show()