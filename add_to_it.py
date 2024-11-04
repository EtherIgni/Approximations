import numpy as np
from copy import deepcopy
from tools import fitting,distributions,initial_estimates
import matplotlib.pyplot as plt
from models import basic
from rmatrix import Particle
import numpy as np
import time
import sys, os

data=np.genfromtxt("sorted runs svd.txt",float,delimiter=" ")
new_data=np.empty((data.shape[0],data.shape[1]+6))
data_end_point=data.shape[1]
new_data[:,:data_end_point]=data

separation_energy=float(7.5767E6) #ev
resonance_distance=600 #ev
resonance_avg_separation=8 #ev
gamma_variance=float(32E-3) #ev
neutron_variance=float(452.5E-3) #ev
first_excited_state=float(6.237E3) #ev
energy_grid_buffer=20 #ev

for index in range(data.shape[0]):
    model=basic.base_reaction()

    resonance_gap=data[index,0]

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
    reduced_width_amplitudes = [data[index,0], data[index,1]]
    model.set_elastic_channel(J,pi,ell,radius,reduced_width_amplitudes)

    J = 3
    pi = 1  # positive parity
    ell = 0 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [data[index,2], data[index,3]]
    excitation = 0  # the product is left in the ground state 
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    J = 3
    pi = 1  # positive parity
    ell = 0 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [data[index,4], data[index,5]]
    excitation = first_excited_state  # the product is left in the 1st ex state at 0.5MeV
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    res_energies=[resonance_distance,resonance_distance+resonance_gap]
    energy_grid=np.linspace(res_energies[0]-energy_grid_buffer,res_energies[1]+energy_grid_buffer,1001)
    model.set_resonance_energies(res_energies)
    model.set_energy_grid(energy_grid)

    model.establish_spin_group()
    
    new_data[index,data_end_point]=model.get_elastic_channel().calc_penetrability(separation_energy-first_excited_state)
    new_data[index,data_end_point+1]=model.get_elastic_channel().calc_penetrability(separation_energy)
    new_data[index,data_end_point+2]=model.get_capture_channels()[0].calc_penetrability(separation_energy-first_excited_state)
    new_data[index,data_end_point+3]=model.get_capture_channels()[0].calc_penetrability(separation_energy)
    new_data[index,data_end_point+4]=model.get_capture_channels()[1].calc_penetrability(separation_energy-first_excited_state)
    new_data[index,data_end_point+5]=model.get_capture_channels()[1].calc_penetrability(separation_energy)

np.savetxt("added runs svd.txt",new_data)