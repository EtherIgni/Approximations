file_path="Approximations/run_data/test_data_2/large_res_gap.txt"

import numpy             as np
import matplotlib.pyplot as plt
import scipy.optimize    as opt
from   tqdm          import tqdm

from Approximations.models.math_models.R_matrix_new import RMInterface
from Approximations.rmatrix.base.particles          import Particle



def solve_R_Matrix(energy_long,Γ1,Γ2,Γ3,Γ4):
    energy_grid=energy_long[:energy_length]
    
    num_channels   = 3
    num_levels     = 2
    
    k            = np.zeros((energy_length, num_channels, num_channels))
    k[:,0,0]     = (elastic_const*compound_nucleons*np.sqrt(energy_grid))/(compound_nucleons+1)
    for i in range(1,num_channels):
        k[:,i,i] = (separation_energy+energy_grid-excitation_energies[i-1])/(hbar*c)

    ρ = k*elastic_radius*ρ_scale

    Γ = np.array([[Γ1,Γ3,0],[Γ2,0,Γ4]])
    Γ = np.repeat(Γ[np.newaxis,:,:], energy_length, 0)

    P          = np.zeros((energy_length, num_channels, num_channels))
    P[:,0,0]   = ρ[:,0,0]
    P[:,1:,1:] = np.power(ρ[:,1:,1:],2*capture_ell+1)
    P          = P.astype(np.complex64)

    L = 1j*P

    P_root = np.sqrt(P)

    O = Γ@L@np.transpose(Γ, [0,2,1])

    E1 = np.diag(resonance_energies)
    E1 = np.repeat(E1[np.newaxis,:,:],   energy_length,  0)
    E2 = np.eye(num_levels)
    E2 = np.repeat(E2[np.newaxis,:,:],   energy_length,  0)
    E2 = energy_grid.reshape(energy_length,1,1)*E2
    E  = E1-E2
    E  = E.astype(np.complex64)

    A = E-O

    A_inv = np.linalg.inv(A)

    Q = np.transpose(Γ, [0,2,1])@A_inv@Γ

    I = np.eye(num_channels)
    I = I.astype(np.complex64)
    I = np.repeat(I[np.newaxis,:,:],   energy_length,  0)

    W = I + 2j*P_root@Q@P_root

    Ω = I[0]*np.repeat(np.exp(-1j*np.diagonal(ρ,axis1=1,axis2=2))[:,:,np.newaxis],num_channels,2)
    Ω = Ω.astype(np.complex64)

    U = Ω@W@Ω

    k2 = np.power(k[:,0,0],2)

    XS          = np.zeros((energy_length, num_channels))
    XS[:,0]     = (10**24 * np.pi/k2 * (1- 2*U[:,0,0].real + np.conjugate(U[:,0,0])*U[:,0,0])).real
    for i in range(1,num_channels):
        XS[:,i] = (10**24 * np.pi/k2 * np.conjugate(U[:,0,i])*U[:,0,i]).real

    measures      = np.zeros((energy_length, 2))
    measures[:,0] = np.sum(XS,1)
    measures[:,1] = np.sum(XS[:,1:],1)
    
    measures_combined = np.concat((measures[:,0], measures[:,1]))
    
    return(measures_combined)



hbar             = 6.582119e-16  # eV-s
c                = 2.99792e10    # cm/s
ρ_scale          = 1e-12
elastic_const    = 0.002197e12

neutron_variance = 452.5E-3
gamma_variance   = 32E-3

excitation_energies = np.array([0,
                                6.237e3,
                                136.269e3,
                                152.320e3,
                                301.622e3,
                                337.54e3])
resonance_energies  = np.array([600,
                                650])
separation_energy   = 7.5767E6
capture_radius      = 0.2
capture_ell         = 0
elastic_radius      = 0.2
compound_nucleons   = 182



num_channels   = 7
num_levels     = 2

excitation_energies=excitation_energies[:num_channels-1]

num_iterations = 1
energy_length  = 2000
energy_buffer  = 5
energy_min     = np.min(resonance_energies)-energy_buffer
energy_max     = np.max(resonance_energies)+energy_buffer
energy_grid    = np.linspace(energy_min, energy_max, energy_length)

r_matrix = RMInterface(num_levels, num_channels)
r_matrix.set_Molecular_Information(compound_nucleons,separation_energy)
r_matrix.set_Energy_Grid(energy_grid)
r_matrix.set_Resonance_Energy(resonance_energies)
r_matrix.set_Excitation_Energy(excitation_energies)
r_matrix.set_Elastic_information(neutron_variance, elastic_radius)
r_matrix.set_Capture_Information(gamma_variance, capture_radius, capture_ell)
r_matrix.calc_Cross_Sections()


k            = np.zeros((energy_length, num_channels, num_channels))
k[:,0,0]     = (elastic_const*compound_nucleons*np.sqrt(energy_grid))/(compound_nucleons+1)
for i in range(1,num_channels):
    k[:,i,i] = (separation_energy+energy_grid-excitation_energies[i-1])/(hbar*c)
print("k  :", k.shape)

ρ = k*elastic_radius*ρ_scale
print("ρ  :", ρ.shape)

Γ         = np.zeros((num_iterations, num_levels, num_channels))
Γ[:,:,0]  = np.random.normal(0, neutron_variance, num_iterations*num_levels                 ).reshape((num_iterations, num_levels                ))
Γ[:,:,1:] = np.random.normal(0, gamma_variance,   num_iterations*num_levels*(num_channels-1)).reshape((num_iterations, num_levels, num_channels-1))
Γ         = np.repeat(Γ[:,np.newaxis,:,:], energy_length, 1)
print("Γ  :", Γ.shape)

P          = np.zeros((energy_length, num_channels, num_channels))
P[:,0,0]   = ρ[:,0,0]
P[:,1:,1:] = np.power(ρ[:,1:,1:],2*capture_ell+1)
P          = P.astype(np.complex64)
P          = np.repeat(P[np.newaxis,:,:,:], num_iterations, 0)
print("P  :", P.shape)

L = 1j*P
print("L  :", L.shape)

P_root = np.sqrt(P)
print("P½ :", P_root.shape)

O = Γ@L@np.transpose(Γ, [0,1,3,2])
print("O  :", O.shape)

E1 = np.diag(resonance_energies)
E1 = np.repeat(E1[np.newaxis,:,:],   energy_length,  0)
E2 = np.eye(num_levels)
E2 = np.repeat(E2[np.newaxis,:,:],   energy_length,  0)
E2 = energy_grid.reshape(energy_length,1,1)*E2
E  = E1-E2
E  = E.astype(np.complex64)
E  = np.repeat(E[np.newaxis,:,:,:], num_iterations, 0)
print("E  :", E.shape)

A = E-O
print("A  :", A.shape)

A_inv = np.linalg.inv(A)
print("A⁻¹:", A_inv.shape)

Q = np.transpose(Γ, [0,1,3,2])@A_inv@Γ
print("Q  :", Q.shape)

I = np.eye(num_channels)
I = I.astype(np.complex64)
I = np.repeat(I[np.newaxis,:,:],   energy_length,  0)
I = np.repeat(I[np.newaxis,:,:,:], num_iterations, 0)
print("I  :", I.shape)

W = I + 2j*P_root@Q@P_root
print("W  :", W.shape)

Ω = I[0]*np.repeat(np.exp(-1j*np.diagonal(ρ,axis1=1,axis2=2))[:,:,np.newaxis],num_channels,2)
Ω = Ω.astype(np.complex64)
Ω = np.repeat(Ω[np.newaxis,:,:,:], num_iterations, 0)
print("Ω  :", Ω.shape)

U = Ω@W@Ω
print("U  :", U.shape)

k2 = np.power(k[:,0,0],2)
k2 = np.repeat(k2[np.newaxis,:], num_iterations, 0)
print("K² :", k2.shape)

XS            = np.zeros((num_iterations, energy_length, num_channels))
XS[:,:,0]     = (10**24 * np.pi/k2 * (1- 2*U[:,:,0,0].real + np.conjugate(U[:,:,0,0])*U[:,:,0,0])).real
for i in range(1,num_channels):
    XS[:,:,i] = (10**24 * np.pi/k2 * np.conjugate(U[:,:,0,i])*U[:,:,0,i]).real
print("XS :", XS.shape)

measures        = np.zeros((num_iterations, energy_length, 2))
measures[:,:,0] = np.sum(XS,2)
measures[:,:,1] = np.sum(XS[:,:,1:],2)
print("Mes:", measures.shape)

measures_combined = np.concat((measures[:,:,0], measures[:,:,1]),1)
print("Mes Combined:", measures_combined.shape)
print()

# iteration_to_show = np.random.randint(0, num_iterations)
# print("Iteration #:",iteration_to_show)
# print()



energy_long=np.concat((energy_grid,energy_grid))

Γ_fit = np.zeros((num_iterations, num_levels, num_levels+1))

for i in tqdm(range(num_iterations),
              desc="Fitting Data",
              ncols=100,
              smoothing=0):
    fitted_params, ___ = opt.curve_fit(solve_R_Matrix, energy_long, measures_combined[i])
    Γ_fit[i]           = np.array([[fitted_params[0],fitted_params[2],0],
                                   [fitted_params[1],0,fitted_params[3]]])
    
    text_data=np.array2string(fitted_params,separator=" ",max_line_width=100000)[1:-1]+"|"
    for j in range(Γ.shape[2]):
        text_data=text_data+np.array2string(Γ[i,0,j],separator=" ",max_line_width=100000)[1:-1]+","
    text_data=text_data[:-1]+"\n"
    # with open(file_path, "a") as f:
    #     f.write(text_data)

# print()
# print("True Gamma Values:")
# print(Γ[iteration_to_show,0])
# print()

# print("Fitted Gamma Values:")
# print(Γ_fit[iteration_to_show])
# print()

# measures_RM = solve_R_Matrix(energy_long,
#                              Γ_fit[iteration_to_show,0,0],
#                              Γ_fit[iteration_to_show,1,0],
#                              Γ_fit[iteration_to_show,0,1],
#                              Γ_fit[iteration_to_show,1,2])
# measures_RM=measures_RM.reshape((energy_length,2),order="F")

# print("Final Error:",np.sum(np.power(measures-measures_RM,2)))



# fig,axes = plt.subplots(3)

# axes[0].plot(energy_grid, measures[iteration_to_show,:,0],                          label="Full Total XS")
# axes[0].plot(energy_grid, measures_RM[:,0],                                         label="R-M Total XS")
# axes[0].legend()
# axes[0].set_xlim(np.min(energy_grid),np.max(energy_grid))
# axes[0].set_ylim(bottom=0)

# axes[1].plot(energy_grid, measures[iteration_to_show,:,1],                          label="Full Gamma XS")
# axes[1].plot(energy_grid, measures_RM[:,1],                                         label="R-M Gamma XS")
# axes[1].legend()
# axes[1].set_xlim(np.min(energy_grid),np.max(energy_grid))
# axes[1].set_ylim(bottom=0)

# axes[2].plot(energy_grid, np.abs(measures[iteration_to_show,:,0]-measures_RM[:,0]), label="Errors Total XS")
# axes[2].plot(energy_grid, np.abs(measures[iteration_to_show,:,1]-measures_RM[:,1]), label="Errors Gamma XS")
# axes[2].legend()
# axes[2].set_xlim(np.min(energy_grid),np.max(energy_grid))
# axes[2].set_ylim(bottom=0)

# plt.show()