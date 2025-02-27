import numpy as np
import matplotlib.pyplot as plt
from Approximations.models.problem_container import Problem
import Approximations.tools.distributions as Distributions

from scipy.optimize import curve_fit 

molecular_information   = {"Incident Name":     "n",
                           "Incident Nucleons":  0,
                           "Incident Protons":   1,
                           "Departing Name":     "g",
                           "Departing Nucleons": 0,
                           "Departing Protons":  0,
                           "Compound Name":      "181Ta",
                           "Compound Nucleons":  181,
                           "Compound Protons":   71}

interaction_information = {"Separation Energy":         float(7.5767E6),
                           "Gamma Variance":            float(32E-3),
                           "Neutron Variance":          float(452.5E-3),
                           "Number Levels":             2,
                           "Resonance Distance":        600,
                           "Resonance Average Spacing": 8}

model_information       = {"Energy Grid Size":   1001,
                           "Energy Grid Buffer": 20}

fitting_parameters      = {"Iteration Limit":        1000,
                           "Improvement Threshold":  0.1,
                           "Initial Priority":       float(10E6),
                           "Priority Multiplier":    1.5,
                           "Priority Minimum":       float(10E-8),
                           "Priority Maximum":       float(10E16)}

selections              = {"Data Model": 1,
                           "Fit Model":  1,
                           "Fit Method": 2}
                
all_excited_states      = np.array([0,
                                    float(6.237E3),
                                    float(136.269E3),
                                    float(152.320E3),
                                    float(301.622E3),
                                    float(337.54E3)])

number_states=6

interaction_information["Excited States"]=all_excited_states[:number_states]

num_levels   = interaction_information["Number Levels"]
num_channels = len(interaction_information["Excited States"])+1

num_iterations=1000

gamma_matrices=np.zeros((num_iterations,num_levels,num_channels))
resonance_gaps=np.zeros((num_iterations,2))

for i in range(num_iterations):
    problem      = Problem(molecular_information,
                           interaction_information,
                           model_information,
                           fitting_parameters,
                           selections)

    gamma_matrices[i]=problem.data_model.math_model.get_Gamma_Matrix()
    resonance_gaps[i]=problem.interaction_information["Resonance Levels"]






num_bins=50
val_1=gamma_matrices.shape[1]
val_2=gamma_matrices.shape[2]
fig,ax=plt.subplots(val_1,val_2)
fig.set_figheight(val_1*5)
fig.set_figwidth(val_2*7)
fig.suptitle("True Gamma Array")

for a in range(val_1):
    for b in range(val_2):
        if(val_2==1):
            axis=ax
        elif(val_1==1):
            axis=ax[a]
        else:
            axis=ax[a,b]
        # try:
        selected_data=gamma_matrices[:,a,b]
        selected_std=np.std(selected_data)
        selected_bins=np.linspace(-3*selected_std,3*selected_std,num_bins)
        counts,bins=np.histogram(selected_data,bins=num_bins,density=True)
        axis.hist(bins[:-1], bins, weights=counts,color="cyan",alpha=0.75,label="Selected runs")
        def Gauss(x, A, B):
            y = A*np.exp(-1*B*x**2)
            return y
        parameters, covariance = curve_fit(Gauss, bins[:-1], counts)
        fit_y = Gauss(bins[:-1], parameters[0], parameters[1])
        axis.plot(bins[:-1],fit_y,color="black",label="Selected Gaussian",zorder=-1)
        axis.set_ylabel("Relative Density")
        axis.set_xlabel("Gamma Matrix Value")
plt.show()

fig,axes=plt.subplots(1,2)
for idx,axis in enumerate(axes):
    axis.hist(resonance_gaps[:,idx],50,color="Red")
plt.show()