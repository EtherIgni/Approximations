import numpy as np

from Approximations.rmatrix.base.particles import Particle
from Approximations.tools import distributions

def create_leveled_model(compound_name,
                         N,
                         Z,
                         separation_energy,
                         resonance_distance,
                         resonance_avg_separation,
                         gamma_variance,
                         neutron_variance,
                         excited_states,
                         energy_grid_buffer,
                         energy_grid_size,
                         model_type):
    number_levels=len(excited_states)
    model=model_type(len(excited_states)+1,len(excited_states))

    resonance_gaps=distributions.sample_wigner_invCDF(number_levels-1)*resonance_avg_separation

    neutron = Particle('n',1,0)
    gamma = Particle('g',0,0)
    target = Particle(compound_name,N,Z)
    compound = Particle(compound_name, N,Z, Sn=separation_energy)
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