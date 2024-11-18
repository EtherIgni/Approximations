import numpy as np
import matplotlib.pyplot as plt

path_prefix="images/generation results/diag/"

data=np.genfromtxt("added runs.txt",float,delimiter=" ")
cut_off_value=10
cut_off_point=np.min((np.sqrt(data[:,11])>cut_off_value).nonzero())
print(cut_off_point,cut_off_point/data.shape[0])

results=data[:cut_off_point,11]
iterations=data[:cut_off_point,12]
gamma_matrices=data[:cut_off_point,7:11]
true_matrices=data[:cut_off_point,1:7]
resonance_separations=data[:cut_off_point,0]
penetrabilities=data[:cut_off_point,13:]







num_bins=30
plt.title("Resonance Separation Distribution")
plt.xlabel("Resonance Separation Value [eV]")
plt.ylabel("Frequency")
plt.hist(resonance_separations-600,num_bins,color="black",density=True)
plt.savefig(path_prefix+"Separation Values.png")
# plt.show()
plt.close()



fig,ax=plt.subplots(1,3)
fig.set_figheight(10)
fig.set_figwidth(20)
fig.suptitle("Gamma Penetrability Test")

titles=["Neutron Channel","Gamma Channel 1","Gamma Channel 2"]
for idx in range(3):
    vals=2*penetrabilities[:,idx*2]*np.power(true_matrices[:,idx*2],2)*np.sign(true_matrices[:,idx*2])+2*penetrabilities[:,idx*2+1]*np.power(true_matrices[:,idx*2+1],2)*np.sign(true_matrices[:,idx*2+1])
    ax[idx].hist(vals,color="orange",density=True)
    ax[idx].text(((ax[idx].get_xlim()[1]-ax[idx].get_xlim()[0])*1/3)+ax[idx].get_xlim()[0], ((ax[idx].get_ylim()[1]-ax[idx].get_ylim()[0])*6/7)+ax[idx].get_ylim()[0],
                    'Variance: {br:.6f}'.format(br=np.std(vals)), style='italic', bbox={
                    'facecolor': 'grey', 'alpha': 1, 'pad': 10})
    ax[idx].set_title(titles[idx])
    ax[idx].set_ylabel("Frequency")
    ax[idx].set_xlabel("Big Gamma Value [eV]")

plt.savefig(path_prefix+"Penetrability Plot.png")
# plt.show()
plt.close()



num_bins=30
fig,ax=plt.subplots(2,3)
fig.set_figheight(10)
fig.set_figwidth(20)
fig.suptitle("True Gamma Matrix Value Distribution")

titles=[["Neutron 1","Neutron 2"],
        ["First Gamma 1","First Gamma 2"],
        ["Second Gamma 1","Second Gamma 2"]]
for col in range(3):
    for row in range(2):
        std=np.std(true_matrices[:,col*2+row])
        bins=np.linspace(-3*std,3*std,num_bins)
        ax[row,col].hist(true_matrices[:,col*2+row],bins=bins,color="blue",density=True)
        ax[row,col].set_ylabel("Frequency")
        ax[row,col].set_xlabel("matrix value")
        ax[row,col].set_title(titles[col][row])
        ax[row,col].text(((ax[row,col].get_xlim()[1]-ax[row,col].get_xlim()[0])*1/3)+ax[row,col].get_xlim()[0], ((ax[row,col].get_ylim()[1]-ax[row,col].get_ylim()[0])*6/7)+ax[row,col].get_ylim()[0],
                        'Variance: {br:.6f}'.format(br=std**2), style='italic', bbox={
                        'facecolor': 'grey', 'alpha': 1, 'pad': 10})

plt.savefig(path_prefix+"True Plot.png")
# plt.show()
plt.close()



num_bins=30
fig,ax=plt.subplots(2,3)
fig.set_figheight(10)
fig.set_figwidth(20)
fig.suptitle("Fitted Gamma Matrix Value Distribution")

titles=[["Neutron 1","Neutron 2"],
        ["First Gamma 1","First Gamma 2"],
        ["Second Gamma 1","Second Gamma 2"]]
row_col_conversion=[(0,0),(1,0),(0,1),(1,2)]
for index,row_col_set in enumerate(row_col_conversion):
    row,col=row_col_set
    std=np.std(gamma_matrices[:,index])
    bins=np.linspace(-3*std,3*std,num_bins)
    ax[row,col].hist(gamma_matrices[:,index],bins=bins,color="red",density=True)
    ax[row,col].set_ylabel("Frequency")
    ax[row,col].set_xlabel("matrix value")
    ax[row,col].set_title("neutron 1")
    ax[row,col].text(((ax[row,col].get_xlim()[1]-ax[row,col].get_xlim()[0])*1/3)+ax[row,col].get_xlim()[0],
                        ((ax[row,col].get_ylim()[1]-ax[row,col].get_ylim()[0])*6/7)+ax[row,col].get_ylim()[0],
                    'Variance: {br:.6f}'.format(br=std**2), style='italic', bbox={
                    'facecolor': 'grey', 'alpha': 1, 'pad': 10})
        
plt.savefig(path_prefix+"Fit Plot.png")
# plt.show()
plt.close()



num_bins=30
fig,ax=plt.subplots(2,3)
fig.set_figheight(10)
fig.set_figwidth(20)
fig.suptitle("Fitted Gamma Matrix Value Distribution")

titles=[["Neutron 1","Neutron 2"],
        ["First Gamma 1","First Gamma 2"],
        ["Second Gamma 1","Second Gamma 2"]]
row_col_conversion=[(0,0),(0,1),(1,0),(1,2)]
for index,row_col_set in enumerate(row_col_conversion):
    row,col=row_col_set
    std=np.std(gamma_matrices[:,index])
    bins=np.linspace(0,3*(std**2),num_bins)
    ax[row,col].hist(np.power(gamma_matrices[:,index],2),bins=bins,color="magenta",density=True)
    ax[row,col].set_ylabel("Frequency")
    ax[row,col].set_xlabel("matrix value")
    ax[row,col].set_title("neutron 1")
        
plt.savefig(path_prefix+"Fit Chi Squared.png")
# plt.show()
plt.close()



num_bins=70
bins=np.linspace(0,10,num_bins)
plt.title("Fit SQE Results")
plt.xlabel("Mean Squared Error")
plt.ylabel("Frequency")
plt.hist(np.sqrt(results),bins=bins,color="green")
plt.savefig(path_prefix+"Fit Results.png")
plt.close()
# plt.show()