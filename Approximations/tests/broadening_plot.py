import matplotlib.pyplot as plt
import numpy as np
import os

cross_section_type = 1

differential_grid = np.load(os.getcwd()+"/Approximations/tests/saved_differential_grid.npy")[:,:,cross_section_type].T

temperature_spread = np.linspace(0,300,101)[1:]
energy_grid = np.linspace(210, 240, 600)

X,Y=np.meshgrid(temperature_spread,energy_grid)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax.plot_surface(X, Y, differential_grid, cmap='viridis', linewidth=0, antialiased=False)
# ax.scatter(X, Y, zs=differential_grid)
# plt.show()


plt.plot(temperature_spread,differential_grid[100])
plt.show()
plt.plot(temperature_spread,differential_grid[80])
plt.show()
plt.plot(temperature_spread,differential_grid[300])
plt.show()
plt.plot(energy_grid,differential_grid[:,0])
plt.show()
plt.plot(energy_grid,differential_grid[:,10])
plt.show()
plt.plot(energy_grid,differential_grid[:,50])
plt.show()
plt.plot(energy_grid,differential_grid[:,-1])
plt.show()