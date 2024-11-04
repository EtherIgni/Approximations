import numpy as np
data=np.genfromtxt("successful runs 7 vd.txt",float,delimiter=" ")
indexes=np.argsort(data[:,13])
new_data=data[indexes,:]
np.savetxt("sorted runs svd.txt",new_data)