import numpy as np
import matplotlib.pyplot as plt
import os 
# print(os.getcwd())
# from Approximations.models import gamma_SVD_model
# from Approximations.tools import numeric_evaluators

# def test_eval(x):
#     return(np.sum(np.power(x,2)))

# print(test_eval(np.array([2,2])))
# print(numeric_evaluators.gradient(np.array([2,2],float),test_eval,0.0000000001))

# print("a b ".split(" ")[:-1])

# with open("Approximations/test_file.txt","w") as file:
#     file.write("")

# print(np.max(np.linalg.svd(np.array([[3,6],[7,2]]))[1]))

# A = np.array([1,2,3])[None,:]
# print(isinstance(A,np.ndarray))

num_vals=10
padding=int(num_vals/2)

A = np.linspace(10,20,num_vals+1)[1:]
B = np.zeros((num_vals,2))
B[:,0] = A
B[:,1] = A*2

print(A)
print()
print(B)
print()
C=np.concat([2*A[0]-np.flip(A[1:padding+1]), A, 2*A[-1]-np.flip(A[-padding-1:-1])])
print(C)
print()
D=np.concat([B[1:padding+1][::-1,0],B,B[-padding-1:-1][::-1,0]])
print(D)
print(D.shape)
print()

# left_A = 2*A[0]  - A[1:padding+1][::-1]
# right_A = 2*A[-1] - A[-padding-1:-1][::-1]
# padded_A = np.concatenate([left_A, A, right_A])
# print(padded_A)
# print()

# left_A = A[1:padding+1][::-1]
# right_A = A[-padding-1:-1][::-1]
# padded_A = np.concatenate([left_A, A, right_A])
# print(padded_A)

# class test():
#     def __init__(self):
#         self.boltz = 1
#         self.target_nucleons = 1
#         self.incident_nucleons = 1
#         num_spaces = 100
#         self.energy_grid = np.linspace(0,100,num_spaces)
#         self.cross_section = np.zeros((num_spaces,2))
#         self.cross_section[:,0] = np.sin(self.energy_grid/100*np.pi)
#         self.cross_section[:,1] = np.sin(self.energy_grid/100*np.pi)
#         self.doppler_Broaden()

#     def doppler_Broaden(self):
        
        
#         doppler_width = np.sqrt(4*self.boltz*self.incident_nucleons*10*self.energy_grid[:,None]/self.target_nucleons)
                
#         energy_factor = np.sqrt(self.energy_grid[None,:]*self.energy_grid[:,None])
#         first_exponential  = np.exp(-4*np.power(self.energy_grid[:,None]-energy_factor,2)/np.power(doppler_width,2))
#         second_exponential = np.exp(-4*np.power(self.energy_grid[:,None]+energy_factor,2)/np.power(doppler_width,2))
#         kernel = (first_exponential-second_exponential) * np.sqrt(self.energy_grid[:,None]/self.energy_grid[None,:])
#         integrand = kernel[:,:,None] * self.cross_section[None,:,:]
#         integral  = np.trapezoid(integrand,self.energy_grid[None,:,None],axis=1)

#         self.cross_section = integral/(doppler_width[:,0, None]*np.sqrt(np.pi))
        
# test_class = test()
# plt.plot(test_class.energy_grid, test_class.cross_section[:,0])
# plt.show()

# A = np.linspace(0,1,1000)
# min_energy=0
# max_energy=100
# targets = np.array([25,75])
# strengths=np.array([5,5])
# widths=np.array([0.05,0.05])
# def cluster_map(num_points, min_energy, max_energy, targets, strengths, widths):
#     uniform_dist = np.linspace(0,1,num_points)
#     clustering_targets = (targets - min_energy)/(max_energy - min_energy)

#     cluster_weights = np.ones_like(uniform_dist)
#     for idx in range(len(clustering_targets)):
#         cluster_weights += strengths[idx] * np.exp(-((uniform_dist - clustering_targets[idx])**2)/(2*widths[idx]**2))
#     cluster_weights = 1/cluster_weights

#     cumulative_weights = np.cumsum(cluster_weights)
#     cumulative_weights = (cumulative_weights - cumulative_weights[0])/(cumulative_weights[-1] - cumulative_weights[0])

#     return min_energy + (max_energy - min_energy)*cumulative_weights

# plt.scatter(A,cluster_map_1(1000,min_energy,max_energy,targets,strengths,widths))
# # plt.scatter(cluster_map_2(1000,min_energy,max_energy,targets,strengths,widths),A)
# plt.show()