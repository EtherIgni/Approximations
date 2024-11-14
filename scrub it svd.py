import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt("added runs svd.txt",float,delimiter=" ")
cut_off_value=200
cut_off_point=np.min((np.sqrt(data[:,13])>cut_off_value).nonzero())
# cut_off_value=np.max(data[:,13])
# cut_off_point=data.shape[0]
print(cut_off_point,cut_off_point/data.shape[0])

results=data[:cut_off_point,13]
iterations=data[:cut_off_point,14]
svd_vals=data[:cut_off_point,7:13]
true_matrices=data[:cut_off_point,1:7]
resonance_separations=data[:cut_off_point,0]
penetrabilities=data[:cut_off_point,15:]
gamma_matrices=np.zeros((cut_off_point,2,3))
svd_matrix=np.zeros((3,3,cut_off_point))
svd_matrix[0,0,:]=svd_vals[:,0]
svd_matrix[0,1,:]=svd_vals[:,1]
svd_matrix[0,2,:]=svd_vals[:,2]
svd_matrix[1,0,:]=svd_vals[:,3]
svd_matrix[1,2,:]=svd_vals[:,4]
svd_matrix[2,2,:]=svd_vals[:,5]
for i in range(cut_off_point):
    U=svd_matrix[:2,0,i][:,None]
    S=svd_matrix[0,1,i]
    Vh=svd_matrix[:3,2,i][None]
    gamma_matrix=(U@Vh)*S
    gamma_matrices[i]=gamma_matrix




num_bins=30
plt.title("Resonance Separation Distribution")
plt.xlabel("Resonance Separation Value [eV]")
plt.ylabel("Frequency")
plt.hist(resonance_separations-600,num_bins,color="black",density=True)
plt.savefig("images/generation results/svd/Separation Values.png")
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

plt.savefig("images/generation results/svd/Penetrability Plot.png")
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

plt.savefig("images/generation results/svd/True Plot.png")
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
row_col_conversion=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
for row in range(2):
    for col in range(3):
        if((row,col) in row_col_conversion):
            std=np.std(gamma_matrices[:,row,col])
            if(std>0):
                bins=np.linspace(-3*std,3*std,num_bins)
            else:
                bins=np.linspace(-1,1,num_bins)
            ax[row,col].hist(gamma_matrices[:,row,col],bins=bins,color="red",density=True)
            ax[row,col].set_ylabel("Frequency")
            ax[row,col].set_xlabel("matrix value")
            ax[row,col].set_title(titles[col][row])
            ax[row,col].text(((ax[row,col].get_xlim()[1]-ax[row,col].get_xlim()[0])*1/3)+ax[row,col].get_xlim()[0],
                                ((ax[row,col].get_ylim()[1]-ax[row,col].get_ylim()[0])*6/7)+ax[row,col].get_ylim()[0],
                            'Variance: {br:.6f}'.format(br=std**2), style='italic', bbox={
                            'facecolor': 'grey', 'alpha': 1, 'pad': 10})
        else:
            fig.delaxes(ax[row,col])
        
plt.savefig("images/generation results/svd/Fit Plot.png")
# plt.show()
plt.close()



num_bins=30
fig,ax=plt.subplots(3,3)
fig.set_figheight(14)
fig.set_figwidth(20)
fig.suptitle("SVD Fitted Value Distribution")

titles=[["U 0","U 1"],
        ["S"],
        ["Vh 0","Vh 1","Vh 2"]]
row_col_conversion=[(0,0),(0,1),(0,2),(1,0),(1,2),(2,2)]
for row in range(3):
    for col in range(3):
        if((row,col) in row_col_conversion):
            distribution=svd_matrix[row,col,:]
            std=np.std(distribution)
            if(std>0):
                bins=np.linspace(-3*std,3*std,num_bins)
            else:
                bins=np.linspace(-1,1,num_bins)
            ax[row,col].hist(distribution,bins=bins,color="cyan",density=True)
            ax[row,col].set_ylabel("Frequency")
            ax[row,col].set_xlabel("matrix value")
            ax[row,col].set_title(titles[col][row])
            ax[row,col].text(((ax[row,col].get_xlim()[1]-ax[row,col].get_xlim()[0])*1/3)+ax[row,col].get_xlim()[0],
                                ((ax[row,col].get_ylim()[1]-ax[row,col].get_ylim()[0])*6/7)+ax[row,col].get_ylim()[0],
                            'Variance: {br:.6f}'.format(br=std**2), style='italic', bbox={
                            'facecolor': 'grey', 'alpha': 1, 'pad': 10})
        else:
            fig.delaxes(ax[row,col])

plt.savefig("images/generation results/svd/SVD Matrix Plot.png")
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
row_col_conversion=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
for index,row_col_set in enumerate(row_col_conversion):
    row,col=row_col_set
    ax[row,col].hist(np.power(gamma_matrices[:,row,col],2),bins=num_bins,color="magenta",density=True)
    ax[row,col].set_ylabel("Frequency")
    ax[row,col].set_xlabel("matrix value")
    ax[row,col].set_title(titles[col][row])
        
plt.savefig("images/generation results/svd/Fit Chi Squared.png")
# plt.show()
plt.close()



num_bins=70
bins=np.linspace(0,cut_off_value,num_bins)
plt.title("Fit SQE Results")
plt.xlabel("Mean Squared Error")
plt.ylabel("Frequency")
plt.hist(np.sqrt(results),bins=bins,color="green")
plt.savefig("images/generation results/svd/Fit Results.png")
plt.close()
# plt.show()