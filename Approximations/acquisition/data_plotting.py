import numpy             as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import gamma

from Approximations.tools.gamma_tools import reshape_RM

num_levels   = 2
num_channels = 3
num_bins     = 500

file_path="Approximations/run_data/breaking/random_even_6_channels.txt"
plot_path="Approximations/images/generation results/breaking/"
plot_title="random_even_6_channels.png"


with open(file_path,"r") as file:
    lines=file.readlines()
    num_lines=len(lines)
    fitted_gamma_data=np.zeros((num_lines,num_levels,num_channels))
    for idx, line in enumerate(lines):
        data=line.split("|")
        gamma_vector=np.fromstring(data[0],sep=" ")
        fitted_gamma_data[idx]=reshape_RM(gamma_vector,num_levels)



fitted_gamma_data=fitted_gamma_data[np.abs(fitted_gamma_data[:,0,0])<1.5]
fitted_gamma_data=fitted_gamma_data[np.abs(fitted_gamma_data[:,1,0])<1.5]
fitted_gamma_data=fitted_gamma_data[np.abs(fitted_gamma_data[:,0,1])<0.2]
fitted_gamma_data=fitted_gamma_data[np.abs(fitted_gamma_data[:,1,2])<0.2]

squared_gamma_data=np.power(fitted_gamma_data,2)



fig, axes = plt.subplots(num_levels, num_channels)
fig.set_figheight(num_levels*5)
fig.set_figwidth(num_channels*7)
fig.suptitle("Fitted Gamma Values")

for idx in range(num_levels*2):
    if(idx<num_levels):
        a = idx
        b = 0
    else:
        a = idx-num_levels
        b = a+1
    axis         = axes[a,b]
    data         = squared_gamma_data[:,a,b]
    counts, bins = np.histogram(data, bins=num_bins, density=True)
    axis.hist(bins[:-1], bins, weights=counts,color="red",label="Data")
    fitted_params=chi2.fit(squared_gamma_data[:,a,b],floc=0)
    x_vals=np.linspace(0,bins[-11],2000)
    fit_example=chi2.pdf(x_vals,fitted_params[0],fitted_params[1],fitted_params[2])
    axis.plot(x_vals,fit_example,color="black",label="X² fit")
    
    axis.set_xlim(bins[0],bins[-1])
    axis.set_ylim(bottom=0)
    axis.legend()
    
    axis.text(0.075, 0.96,'DF: {br:.6f}'.format(br=fitted_params[0]),style='italic',transform=axis.transAxes,va="top",ha="left",
            bbox={'facecolor':'white','alpha':1,'pad':10})

fig.delaxes(axes[0,2])
fig.delaxes(axes[1,1])

plt.savefig(plot_path+plot_title,dpi=500)

