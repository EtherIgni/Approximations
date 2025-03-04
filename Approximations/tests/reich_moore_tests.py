import numpy             as np
import matplotlib.pyplot as plt
import scipy.optimize    as opt



def plot_Matrix(matrix,title,color):
    num_bins=500
    matrix_shape=matrix.shape
    fig,ax=plt.subplots(matrix_shape[1],matrix_shape[2])
    fig.set_figheight(matrix_shape[1]*5)
    fig.set_figwidth(matrix_shape[2]*7)
    fig.suptitle(title)
    for a in range(matrix_shape[1]):
        for b in range(matrix_shape[2]):
            if(matrix_shape[2]==1):
                axis=ax
            elif(matrix_shape[1]==1):
                axis=ax[a]
            else:
                axis=ax[a,b]
            # try:
            data=matrix[:,a,b]
            counts,bins=np.histogram(data,bins=num_bins,density=True)
            axis.hist(bins[:-1], bins, weights=counts,color=color,label="Data")
            # def Gauss(x, A, B):
            #     y = A*np.exp(-1*B*x**2)
            #     return y
            # parameters, covariance = opt.curve_fit(Gauss, bins[:-1], counts)
            # fit_y = Gauss(bins[:-1], parameters[0], parameters[1])
            # axis.plot(bins[:-1],fit_y,color="black",label="Gaussian Fit",zorder=-1)
    plt.show()



hbar             = 6.582119e-16  # eV-s
c                = 2.99792e10    # cm/s
ρ_scale          = 1e-12
elastic_const    = 0.002197e-12

neutron_variance = 452.5E-3
gamma_variance   = 32E-3

excitation_energies = np.array([0,
                                6.237E3])
resonance_energies  = np.array([600,
                                608])
separation_energy   = 7.5767E6
capture_radius      = 0.2
capture_ell         = 0
elastic_radius      = 0.2
compound_nucleons   = 182



num_channels   = 3
num_levels     = 2

num_iterations = 1
energy_length  = 1000
energy_buffer  = 1
energy_min     = np.min(resonance_energies)-energy_buffer
energy_max     = np.max(resonance_energies)+energy_buffer
energy_grid    = np.linspace(energy_min, energy_max, energy_length)



k            = np.zeros((energy_length, num_channels, num_channels))
k[:,0,0]     = (elastic_const*compound_nucleons*np.sqrt(energy_grid))/(compound_nucleons+1)
for i in range(1,num_channels):
    k[:,i,i] = (separation_energy+energy_grid-excitation_energies[i-1])/(hbar*c)
print("k  : ", k.shape)

ρ = k*elastic_radius*ρ_scale
print("ρ  : ", ρ.shape)

Γ         = np.zeros((num_iterations, num_levels, num_channels))
Γ[:,:,0]  = np.random.normal(0, neutron_variance, num_iterations*num_levels                 ).reshape((num_iterations, num_levels                ))
Γ[:,:,1:] = np.random.normal(0, gamma_variance,   num_iterations*num_levels*(num_channels-1)).reshape((num_iterations, num_levels, num_channels-1))
Γ         = Γ.astype(np.complex64)
Γ         = np.repeat(Γ[:,np.newaxis,:,:], energy_length, 1)
print("Γ  : ", Γ.shape)

P          = np.zeros((energy_length, num_channels, num_channels))
P[:,0,0]   = ρ[:,0,0]
P[:,1:,1:] = np.power(ρ[:,1:,1:],2*capture_ell+1)
P          = P.astype(np.complex64)
P          = np.repeat(P[np.newaxis,:,:,:], num_iterations, 0)
print("P  : ", P.shape)

L = 1j*P
print("L  : ", L.shape)

P_root = np.sqrt(P)
print("P½ : ", P_root.shape)

O = Γ@L@np.transpose(Γ, [0,1,3,2])
print("O  : ", O.shape)

E1 = np.diag(resonance_energies)
E1 = np.repeat(E1[np.newaxis,:,:],   energy_length,  0)
E2 = np.eye(num_levels)
E2 = np.repeat(E2[np.newaxis,:,:],   energy_length,  0)
E2 = energy_grid.reshape(energy_length,1,1)*E2
E  = E1-E2
E  = E.astype(np.complex64)
E  = np.repeat(E[np.newaxis,:,:,:], num_iterations, 0)
print("E  : ", E.shape)

A = E-O
print("A  : ", A.shape)

A_inv = np.linalg.inv(A)
print("A⁻¹: ", A_inv.shape)

Q = np.transpose(Γ, [0,1,3,2])@A_inv@Γ
print("Q  : ", Q.shape)

I = np.eye(num_channels)
I = I.astype(np.complex64)
I = np.repeat(I[np.newaxis,:,:],   energy_length,  0)
I = np.repeat(I[np.newaxis,:,:,:], num_iterations, 0)
print("I  : ", I.shape)

W = I + 2j*P_root@Q@P_root
print("W  : ", W.shape)

Ω = np.exp(-1j*ρ)
Ω = Ω.astype(np.complex64)
Ω = np.repeat(Ω[np.newaxis,:,:,:], num_iterations, 0)
print("Ω  : ", Ω.shape)

U = Ω@W@Ω
print("U  : ", U.shape)

k2 = np.power(k[:,0,0],2)
k2 = np.repeat(k2[np.newaxis,:], num_iterations, 0)
print("K² : ", k2.shape)

XS = 10**24*2*np.pi/k2*(1-U[:,:,0,0].real)
print("XS : ", XS.shape)

iteration_to_show = np.random.randint(0, num_iterations)
print(iteration_to_show)

plt.plot(energy_grid, XS[iteration_to_show])
plt.show()