dir_path     = "/home/aaron/Depo/Nuclear_Research/Approximations/Approximations/run_data/syndat/generated_100_channel_complies"
num_channels = 50






import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import chi2
from ast import literal_eval






final_fit_values = []
with open(dir_path+"/successful_run_data.txt", "r") as file:
    for line in file:
        final_vector = line.replace(" ","").split("|")[1]
        final_vector = np.array(literal_eval(final_vector))
        final_fit_values.append(final_vector)
final_fit_values = np.array(final_fit_values)






data = final_fit_values[:,-1]
data=np.power(data,2)
data=np.sort(data)
data=data/np.mean(data)
centering=np.mean(data)

data = data[np.where(data>0.25)]
data = data[np.where(data<3)]






fig,axis=plt.subplots(1)
fig.set_figheight(15)
fig.set_figwidth(21)

num_bins=100

test_data=np.zeros(len(data))
for i in range(num_channels):
    test_data=test_data+np.power(np.random.normal(0,1,len(test_data)),2)
test_scale=np.mean(test_data)
test_data=test_data/test_scale

n, bins, rects=axis.hist(test_data,bins=num_bins,color="green",alpha=0.75,density=True,label="Expected Shape")
n, bins, rects=axis.hist(data[:int(data.size*0.98)],bins=num_bins,color="purple",alpha=0.75,density=True,label="Gamma Matrix Values")
xlim=axis.get_xlim()
x_vals=np.linspace(0,xlim[1],2000)
fitted_params=chi2.fit(data,floc=0)
fitted_test_params=chi2.fit(test_data,floc=0)
y_vals_1=chi2.pdf(x_vals,fitted_params[0],fitted_params[1],fitted_params[2])
y_vals_test=chi2.pdf(x_vals,fitted_test_params[0],fitted_test_params[1],fitted_test_params[2])
# y_vals_2=chi2.pdf(x_vals,6,0,centering/4)
# y_vals_3=chi_squared(x_vals,num,centering/(2*num))
axis.plot(x_vals,y_vals_1,color="blue")
axis.plot(x_vals,y_vals_test,color="black")
# axis.plot(x_vals,y_vals_2,color="green")
# axis.plot(x_vals,y_vals_3,color="red")
ylim=axis.get_ylim()
xlim=axis.get_xlim()
axis.vlines(centering,0,ylim[1],color="Black")
axis.set_ylim(0,ylim[1])
axis.set_xlim(0,xlim[1])
axis.text(0.03, 0.96,'DF: {br:.6f}'.format(br=fitted_params[0]),style='italic',transform=axis.transAxes,va="top",ha="left",
            bbox={'facecolor':'lightgrey','alpha':1,'pad':10})
axis.text(0.122, 0.9,'DF: {br:.6f}'.format(br=fitted_test_params[0]),style='italic',transform=axis.transAxes,va="top",ha="left",
            bbox={'facecolor':'lightgrey','alpha':1,'pad':10})
axis.legend(prop={'size': 16})
plt.savefig(dir_path+"/distribution_plot.png")



with open(dir_path+"/distribution_results.txt", "w") as file:
    file.write("Expected degrees of freedom: "+str(num_channels)+"\nFitted degrees of freedom: "+str(fitted_params[0]))