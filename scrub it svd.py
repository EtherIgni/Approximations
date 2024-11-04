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
gamma_matrices=np.zeros((cut_off_point,6))
for i in range(cut_off_point):
    svd_matrix=np.array([[svd_vals[i,0],svd_vals[i,1],svd_vals[i,2]],[svd_vals[i,3],0,svd_vals[i,4]],[0,0,svd_vals[i,5]]])
    U=svd_matrix[:2,0][:,None]
    S=svd_matrix[0,1]
    Vh=svd_matrix[:3,2][None]
    gamma_matrix=(U@Vh)*S
    gamma_matrices[i,0]=gamma_matrix[0,0]
    gamma_matrices[i,1]=gamma_matrix[0,1]
    gamma_matrices[i,2]=gamma_matrix[0,2]
    gamma_matrices[i,3]=gamma_matrix[1,0]
    gamma_matrices[i,4]=gamma_matrix[1,1]
    gamma_matrices[i,5]=gamma_matrix[1,2]

print(np.std(gamma_matrices[:,0]))
print(np.std(gamma_matrices[:,1]))
print(np.std(gamma_matrices[:,2]))
print(np.std(gamma_matrices[:,3]))




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
for index,row_col_set in enumerate(row_col_conversion):
    row,col=row_col_set
    std=np.std(gamma_matrices[:,index])
    if(std>0):
        bins=np.linspace(-3*std,3*std,num_bins)
    else:
        bins=np.linspace(-1,1,num_bins)
    ax[row,col].hist(gamma_matrices[:,index],bins=bins,color="red",density=True)
    ax[row,col].set_ylabel("Frequency")
    ax[row,col].set_xlabel("matrix value")
    ax[row,col].set_title("neutron 1")
    ax[row,col].text(((ax[row,col].get_xlim()[1]-ax[row,col].get_xlim()[0])*1/3)+ax[row,col].get_xlim()[0],
                        ((ax[row,col].get_ylim()[1]-ax[row,col].get_ylim()[0])*6/7)+ax[row,col].get_ylim()[0],
                    'Variance: {br:.6f}'.format(br=std**2), style='italic', bbox={
                    'facecolor': 'grey', 'alpha': 1, 'pad': 10})
        
plt.savefig("images/generation results/svd/Fit Plot.png")
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
    ax[row,col].hist(np.power(gamma_matrices[:,index],2),bins=num_bins,color="magenta",density=True)
    ax[row,col].set_ylabel("Frequency")
    ax[row,col].set_xlabel("matrix value")
    ax[row,col].set_title("neutron 1")
        
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