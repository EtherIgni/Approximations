batch_path="/home/aaron/Desktop/Ta-181 6 Levels 2"
output_path="/home/aaron/Desktop/Ta-181 6 Levels 2"
mode=1 #1:Reich-Moore, 2:gamma SVD
cut_off_value=1000
show_ignored_values=True
show_failed_values=False
show_combined_values=True
target_for_penetrability_test=float(32E-3)







import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from scipy.stats import chi2
from scipy.special import gamma

from Approximations.models import gamma_SVD_model,reich_moore_model



if(mode==1):
    folder_name="rm"
elif(mode==2):
    folder_name="svd"
else:
    assert False, "Wrong mode"
if(not(os.path.isdir(output_path))):
    os.mkdir(output_path)


file=open(batch_path+"/successful run data.txt","r")
lines=file.readlines()

successful_run_ids=[]
successful_initial_array_data=[]
successful_final_array_data=[]
successful_initial_gm_matrix_data=[]
successful_final_gm_matrix_data=[]
successful_results_data=[]
successful_iteration_count_data=[]

for line in lines:
    line=line[:-1]
    data=line.split("|")
    successful_run_ids.append(data[0].split(" ")[:-1])
    initial=np.array(data[1].split(" ")[1:-1],float).flatten()
    final=np.array(data[2].split(" ")[1:-1],float).flatten()
    if(mode==1):
        initial_gm=reich_moore_model.reshape(initial)
        final_gm=reich_moore_model.reshape(final)
    elif(mode==2):
        initial_gm=gamma_SVD_model.reshape(initial)
        final_gm=gamma_SVD_model.reshape(final)
    successful_initial_array_data.append(initial)
    successful_final_array_data.append(final)
    successful_initial_gm_matrix_data.append(initial_gm)
    successful_final_gm_matrix_data.append(final_gm)
    successful_results_data.append(data[3].split(" ")[1:-1])
    successful_iteration_count_data.append(data[4].split(" ")[1:])

successful_run_ids=np.array(successful_run_ids,float)
successful_initial_array_data=np.array(successful_initial_array_data,float)
successful_final_array_data=np.array(successful_final_array_data,float)
successful_initial_gm_matrix_data=np.array(successful_initial_gm_matrix_data,float)
successful_final_gm_matrix_data=np.array(successful_final_gm_matrix_data,float)
successful_results_data=np.array(successful_results_data,float)
successful_iteration_count_data=np.array(successful_iteration_count_data,float)



file=open(batch_path+"/model data.txt","r")
lines=file.readlines()

model_run_ids=[]
model_resonance_energy_data=[]
model_penetrability_data=[]
model_comb_penetrability_data=[]
model_true_gamma_data=[]

for line in lines:
    line=line[:-1]
    data=line.split("|")
    model_run_ids.append(data[0].split(" ")[:-1])
    model_resonance_energy_data.append(data[1].split(" ")[1:-1])
    penetrability_matrix=np.zeros((len(model_resonance_energy_data[-1])+1,len(model_resonance_energy_data[-1])+2))
    penetrability_matrix[:,0]=np.array(data[2].split(" ")[1:-1],float)
    penetrability_matrix[:,1:]=np.reshape(np.array(data[3].split(" ")[1:-1],float),(penetrability_matrix.shape[0],penetrability_matrix.shape[1]-1))
    model_penetrability_data.append(penetrability_matrix)
    gamma_matrix=np.reshape(np.array(data[4].split(" ")[1:],float),penetrability_matrix.shape)
    model_true_gamma_data.append(gamma_matrix)

model_run_ids=np.array(model_run_ids,float)
model_resonance_energy_data=np.array(model_resonance_energy_data,float)
model_penetrability_data=np.array(model_penetrability_data,float)
model_true_gamma_data=np.array(model_true_gamma_data,float)



successful_shape=successful_run_ids.shape[0]
successful_resonance_energy_data=np.zeros((successful_shape,model_resonance_energy_data.shape[1]))
successful_penetrability_data=np.zeros((successful_shape,model_penetrability_data.shape[1],model_penetrability_data.shape[2]))
successful_true_gamma_data=np.zeros((successful_shape,model_true_gamma_data.shape[1],model_true_gamma_data.shape[2]))

failed_shape=model_run_ids.shape[0]-successful_run_ids.shape[0]
failed_resonance_energy_data=np.zeros((failed_shape,model_resonance_energy_data.shape[1]))
failed_penetrability_data=np.zeros((failed_shape,model_penetrability_data.shape[1],model_penetrability_data.shape[2]))
failed_true_gamma_data=np.zeros((failed_shape,model_true_gamma_data.shape[1],model_true_gamma_data.shape[2]))

fail_id=0
for idx in range(model_run_ids.shape[0]):
    matches=np.where((successful_run_ids[:,0]==model_run_ids[idx,0])&(successful_run_ids[:,1]==model_run_ids[idx,1]))[0]
    if(len(matches)==1):
        successful_resonance_energy_data[matches[0],:]=model_resonance_energy_data[idx,:]
        successful_penetrability_data[matches[0],:]=model_penetrability_data[idx,:]
        successful_true_gamma_data[matches[0],:]=model_true_gamma_data[idx,:]
    elif(matches.size>1):
        print("Duplicate Run Values")
    else:
        failed_resonance_energy_data[fail_id,:]=model_resonance_energy_data[idx,:]
        failed_penetrability_data[fail_id,:]=model_penetrability_data[idx,:]
        failed_true_gamma_data[fail_id,:]=model_true_gamma_data[idx,:]
        fail_id+=1



sort_indices=np.argsort(successful_results_data,0)
successful_results_data=np.take_along_axis(successful_results_data, sort_indices, axis=0)
cut_off_point=np.min((np.sqrt(successful_results_data.flatten())>cut_off_value).nonzero())
cut_indices=sort_indices[:cut_off_point]
ignored_indices=sort_indices[cut_off_point:]

selected_final_gm_matrix_data=np.take_along_axis(successful_final_gm_matrix_data, cut_indices[:,:,np.newaxis], axis=0)
selected_true_gamma_data=np.take_along_axis(successful_true_gamma_data, cut_indices[:,:,np.newaxis], axis=0)











fig,axis=plt.subplots(1)
fig.set_figheight(15)
fig.set_figwidth(21)

num_bins=100

data=selected_final_gm_matrix_data[:,-6,-6]
data=np.power(data,2)
data=np.sort(data)
cut_off_point=np.min((data.flatten()>1.2).nonzero())
data=data[0:cut_off_point]
data=data/np.mean(data)
centering=np.mean(data)

test_data=np.zeros(len(data))
for i in range(6):
    test_data=test_data+np.power(np.random.normal(0,1,len(test_data)),2)
test_scale=np.mean(test_data)
test_data=test_data/test_scale

n, bins, rects=axis.hist(test_data,bins=num_bins,color="green",alpha=0.75,density=True,label="Expected Shape")
n, bins, rects=axis.hist(data,bins=num_bins,color="purple",alpha=0.75,density=True,label="Gamma Matrix Values")
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
plt.show()
print(fitted_test_params)