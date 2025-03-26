batch_path                    = "/home/aaron/Desktop/batch 6"
output_path                   = "/home/aaron/Desktop/batch 6"
cut_data                      = False
cut_off_value                 = 1000
show_ignored_values           = False
show_failed_values            = False
show_combined_values          = False
target_for_penetrability_test = float(32E-3)








import os
import numpy              as     np
import matplotlib.pyplot  as     plt
import matplotlib.cm      as     cm
from   matplotlib.widgets import Slider
from   scipy.stats        import chi2
from   scipy.special      import gamma

from Approximations.models.fitting_models import reich_moore_model as problemModel






if(not(os.path.isdir(output_path))):
    os.mkdir(output_path)



res_data_file  = open(batch_path+"/successful run data.txt", "r")
res_data_lines = res_data_file.readlines()

successful_run_ids                = []
successful_initial_array_data     = []
successful_final_array_data       = []
successful_initial_gm_matrix_data = []
successful_final_gm_matrix_data   = []
successful_results_data           = []
successful_iteration_count_data   = []

for line in res_data_lines:
    line       = line[:-1] #Cuts the new line off the line
    data       = line.split("|")
    successful_run_ids.append(data[0].split(" ")[:-1])
    initial    = np.array(data[1].split(" ")[1:-1], float).flatten()
    final      = np.array(data[2].split(" ")[1:-1], float).flatten()
    initial_gm = problemModel.reshape(initial, int(len(initial)/2), int(len(initial)/2)+1)
    final_gm   = problemModel.reshape(final,   int(len(initial)/2), int(len(initial)/2)+1)
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



model_data_file  = open(batch_path+"/model data.txt", "r")
model_data_lines = model_data_file.readlines()

model_run_ids                 = []
model_resonance_energy_data   = []
model_penetrability_data      = []
model_comb_penetrability_data = []
model_true_gamma_data         = []

for line in model_data_lines:
    line                       = line[:-1]
    data                       = line.split("|")
    model_run_ids.append(data[0].split(" ")[:-1])
    model_resonance_energy_data.append(data[1].split(" ")[1:-1])

    penetrability_neutron_data = np.array(data[2].split(" ")[1:-1], float)
    penetrability_gamma_data   = np.array(data[3].split(" ")[1:-1], float)
    penetrability_matrix       = np.zeros((penetrability_neutron_data.size,int(penetrability_gamma_data.size/penetrability_neutron_data.size)+1))
    penetrability_matrix[:,0]  = penetrability_neutron_data
    penetrability_matrix[:,1:] = np.reshape(penetrability_gamma_data, (penetrability_matrix.shape[0], penetrability_matrix.shape[1]-1))
    model_penetrability_data.append(penetrability_matrix)
    gamma_matrix               = np.reshape(np.array(data[4].split(" ")[1:], float), (2,6))
    model_true_gamma_data.append(gamma_matrix)

model_run_ids               = np.array(model_run_ids,float)
model_resonance_energy_data = np.array(model_resonance_energy_data,float)
model_penetrability_data    = np.array(model_penetrability_data,float)
model_true_gamma_data       = np.array(model_true_gamma_data,float)



successful_shape                 = successful_run_ids.shape[0]
successful_resonance_energy_data = np.zeros((successful_shape, model_resonance_energy_data.shape[1]))
successful_penetrability_data    = np.zeros((successful_shape, model_penetrability_data.shape[1], model_penetrability_data.shape[2]))
successful_true_gamma_data       = np.zeros((successful_shape, model_true_gamma_data.shape[1], model_true_gamma_data.shape[2]))

failed_shape                     = model_run_ids.shape[0]-successful_run_ids.shape[0]
failed_resonance_energy_data     = np.zeros((failed_shape, model_resonance_energy_data.shape[1]))
failed_penetrability_data        = np.zeros((failed_shape, model_penetrability_data.shape[1], model_penetrability_data.shape[2]))
failed_true_gamma_data           = np.zeros((failed_shape, model_true_gamma_data.shape[1], model_true_gamma_data.shape[2]))

fail_id = 0
for idx in range(model_run_ids.shape[0]):
    matches = np.where((successful_run_ids[:,0]==model_run_ids[idx,0])&(successful_run_ids[:,1]==model_run_ids[idx,1]))[0]
    if(matches.size==1):
        successful_resonance_energy_data[matches[0],:] = model_resonance_energy_data[idx,:]
        successful_penetrability_data[matches[0],:]    = model_penetrability_data[idx,:]
        successful_true_gamma_data[matches[0],:]       = model_true_gamma_data[idx,:]
    elif(matches.size>1):
        print("Duplicate Run Values")
    else:
        failed_resonance_energy_data[fail_id,:] = model_resonance_energy_data[idx,:]
        failed_penetrability_data[fail_id,:]    = model_penetrability_data[idx,:]
        failed_true_gamma_data[fail_id,:]       = model_true_gamma_data[idx,:]
        fail_id += 1



sort_indices            = np.argsort(successful_results_data,0)
successful_results_data = np.take_along_axis(successful_results_data, sort_indices, axis=0)
unmatched_sets          = (np.sqrt(successful_results_data.flatten())>cut_off_value).nonzero()
if(len(unmatched_sets[0])==0 or not(cut_data)):
    cut_off_point = successful_results_data.size
else:
    cut_off_point = np.min((np.sqrt(successful_results_data.flatten())>cut_off_value).nonzero())
print("Cut off index: {:}, Cut off percentage: {:.2f}".format(cut_off_point,100*cut_off_point/successful_results_data.size))
cut_indices     = sort_indices[:cut_off_point]
ignored_indices = sort_indices[cut_off_point:]

ignored_initial_array_data=successful_results_data[cut_off_point:]
ignored_initial_array_data=np.take_along_axis(successful_initial_array_data, ignored_indices, axis=0)
ignored_final_array_data=np.take_along_axis(successful_final_array_data, ignored_indices, axis=0)
ignored_initial_gm_matrix_data=np.take_along_axis(successful_initial_gm_matrix_data, ignored_indices[:,:,np.newaxis], axis=0)
ignored_final_gm_matrix_data=np.take_along_axis(successful_final_gm_matrix_data, ignored_indices[:,:,np.newaxis], axis=0)
ignored_results_data=np.take_along_axis(successful_results_data, ignored_indices, axis=0)
ignored_iteration_count_data=np.take_along_axis(successful_iteration_count_data, ignored_indices, axis=0)
ignored_resonance_energy_data=np.take_along_axis(successful_resonance_energy_data, ignored_indices, axis=0)
ignored_penetrability_data=np.take_along_axis(successful_penetrability_data, ignored_indices[:,:,np.newaxis], axis=0)
ignored_true_gamma_data=np.take_along_axis(successful_true_gamma_data, ignored_indices[:,:,np.newaxis], axis=0)

selected_results_data=successful_results_data[:cut_off_point]
selected_initial_array_data=np.take_along_axis(successful_initial_array_data, cut_indices, axis=0)
selected_final_array_data=np.take_along_axis(successful_final_array_data, cut_indices, axis=0)
selected_initial_gm_matrix_data=np.take_along_axis(successful_initial_gm_matrix_data, cut_indices[:,:,np.newaxis], axis=0)
selected_final_gm_matrix_data=np.take_along_axis(successful_final_gm_matrix_data, cut_indices[:,:,np.newaxis], axis=0)
selected_iteration_count_data=np.take_along_axis(successful_iteration_count_data, cut_indices, axis=0)
selected_resonance_energy_data=np.take_along_axis(successful_resonance_energy_data, cut_indices, axis=0)
selected_penetrability_data=np.take_along_axis(successful_penetrability_data, cut_indices[:,:,np.newaxis], axis=0)
selected_true_gamma_data=np.take_along_axis(successful_true_gamma_data, cut_indices[:,:,np.newaxis], axis=0)






#Resonance Energy Plotting
val_1=int(np.ceil(selected_resonance_energy_data.shape[1]/3))
val_2=int(np.min([selected_resonance_energy_data.shape[1],3]))
fig,ax=plt.subplots(val_1,val_2)
fig.set_figheight(val_1*5)
fig.set_figwidth(val_2*7)
fig.suptitle("Resonance Energy Gap Distributions")

for a in range(val_1):
    for b in range(val_2):
        try:
            if(val_2==1):
                axis=ax
            elif(val_1==1):
                axis=ax[a]
            else:
                axis=ax[a,b]
            if(show_ignored_values):
                axis.hist(failed_resonance_energy_data[:,b+a*3],50,color="black",alpha=0.5,density=True,label="Ignored Runs")
            if(show_failed_values):
                axis.hist(failed_resonance_energy_data[:,b+a*3],50,color="firebrick",alpha=0.5,density=True,label="Failed Runs")
            if(show_combined_values):
                axis.hist(successful_resonance_energy_data[:,b+a*3],50,color="sienna",alpha=0.5,density=True,label="Combined Runs")
            axis.hist(selected_resonance_energy_data[:,b+a*3],50,color="magenta",alpha=0.75,density=True,label="Selected Runs")
            axis.set_ylabel("Relative Density")
            axis.set_xlabel("Resonance Gap Distance")
            if(show_ignored_values | show_failed_values):
                axis.legend()
        except:
            fig.delaxes(ax[a,b])
plt.savefig(output_path+"/Resonance Energy Gap Plot.png")



# #Penetrability Plotting
# num_bins=50
# val_1=selected_penetrability_data.shape[2]
# fig,ax=plt.subplots(1,val_1)
# fig.set_figheight(5)
# fig.set_figwidth(val_1*7)
# fig.suptitle("Resonance Energy Gap Distributions")

# for a in range(val_1):
#     axis=ax[a]
#     # try:
#     if(show_ignored_values):
#         ignored_data=np.sum((ignored_penetrability_data*np.power(ignored_true_gamma_data,2)*np.sign(ignored_true_gamma_data))[:,:,a],1)
#         ignored_std=np.std(ignored_data)
#         ignored_bins=np.linspace(-3*ignored_std,3*ignored_std,num_bins)
#         axis.hist(ignored_data,bins=ignored_bins,color="black",alpha=0.5,density=True,label="Ignored Runs")
#     if(show_failed_values): 
#         failed_data=np.sum((failed_penetrability_data*np.power(failed_true_gamma_data,2)*np.sign(failed_true_gamma_data))[:,:,a],1)
#         failed_std=np.std(failed_data)
#         failed_bins=np.linspace(-3*failed_std,3*failed_std,num_bins)
#         axis.hist(failed_data,bins=failed_bins,color="firebrick",alpha=0.5,density=True,label="Failed Runs")
#     if(show_combined_values):
#         successful_data=np.sum((successful_penetrability_data*np.power(successful_true_gamma_data,2)*np.sign(successful_true_gamma_data))[:,:,a],1)
#         successful_std=np.std(successful_data)
#         successful_bins=np.linspace(-3*successful_std,3*successful_std,num_bins)
#         axis.hist(successful_data,bins=successful_bins,color="sienna",alpha=0.5,density=True,label="Combined Runs")
#     selected_data=np.sum((selected_penetrability_data*np.power(selected_true_gamma_data,2)*np.sign(selected_true_gamma_data))[:,:,a],1)
#     selected_std=np.std(selected_data)
#     selected_bins=np.linspace(-3*selected_std,3*selected_std,num_bins)
#     axis.hist(selected_data,bins=selected_bins,color="orange",alpha=0.75,density=True,label="Selected Runs")
#     axis.text(0.05, 0.93,'Variance: {br:.6f}'.format(br=selected_std**2),style='italic',transform=axis.transAxes,va="top",
#                 bbox={'facecolor':'lightgrey','alpha':1,'pad':10})
#     axis.set_ylabel("Relative Density")
#     axis.set_xlabel("Channel Variance")
#     if(show_ignored_values | show_failed_values):
#             axis.legend()
#     # except:
#     #     fig.delaxes(axis)
# plt.savefig(output_path+"/Penetrability Plot.png")



#True Gamma Array Plotting
num_bins=50
val_1=selected_true_gamma_data.shape[1]
val_2=selected_true_gamma_data.shape[2]
fig,ax=plt.subplots(val_1,val_2)
fig.set_figheight(val_1*5)
fig.set_figwidth(val_2*7)
fig.suptitle("True Gamma Array")
average_penetrability=np.mean(selected_penetrability_data,0)
variances=np.zeros(average_penetrability.shape)

for a in range(val_1):
    for b in range(val_2):
        if(val_2==1):
            axis=ax
        elif(val_1==1):
            axis=ax[a]
        else:
            axis=ax[a,b]
        try:
            selected_data=selected_true_gamma_data[:,a,b]
            selected_std=np.std(selected_data)
            variance=selected_std**2
            selected_bins=np.linspace(-3*selected_std,3*selected_std,num_bins)
            x=np.linspace(selected_bins[0],selected_bins[-1],200)
            gauss=(1/np.sqrt(selected_std*2*np.pi))*np.exp(-0.5*(np.power(x,2)/np.power(selected_std,2)))
            axis.plot(x,gauss,color="black",label="Selected Gaussian",zorder=-1)

            if(show_combined_values):
                successful_data=successful_true_gamma_data[:,a,b]
                successful_std=np.std(successful_data)
                successful_bins=np.linspace(-3*successful_std,3*successful_std,num_bins)
                x=np.linspace(successful_bins[0],successful_bins[-1],200)
                gauss=(1/np.sqrt(successful_std*2*np.pi))*np.exp(-0.5*(np.power(x,2)/np.power(successful_std,2)))
                axis.plot(x,gauss,color="sienna",label="Combined Gaussian",zorder=-1)
                axis.hist(successful_data,bins=successful_bins,color="sienna",alpha=0.5,density=True,label="Combined Runs")
            if(show_ignored_values):
                ignored_data=ignored_true_gamma_data[:,a,b]
                ignored_std=np.std(ignored_data)
                ignored_bins=np.linspace(-3*ignored_std,3*ignored_std,num_bins)
                axis.hist(ignored_data,bins=ignored_bins,color="black",alpha=0.5,density=True,label="Ignored Runs")
            if(show_failed_values):
                failed_data=failed_true_gamma_data[:,a,b]
                failed_std=np.std(failed_data)
                failed_bins=np.linspace(-3*failed_std,3*failed_std,num_bins)
                axis.hist(failed_data,bins=failed_bins,color="firebrick",alpha=0.5,density=True,label="Failed Runs")
            
            axis.hist(selected_data,bins=selected_bins,color="cyan",alpha=0.75,density=True,label="Selected Runs")
            axis.set_ylabel("Relative Density")
            axis.set_xlabel("Gamma Matrix Value")
            axis.text(0.05, 0.93,'Variance: {br:.6f}'.format(br=selected_std**2),style='italic',transform=axis.transAxes,va="top",
                        bbox={'facecolor':'lightgrey','alpha':1,'pad':10})
            if(show_ignored_values | show_failed_values):
                axis.legend()
        except:
            fig.delaxes(axis)
plt.savefig(output_path+"/True Gamma Matrix Values Plot.png")
# penetrability_test_result=np.mean(np.sum(variances*average_penetrability,0)[1:])
# print("Penetrability test relative distance: {:.3f}".format(np.abs(penetrability_test_result-target_for_penetrability_test)/target_for_penetrability_test))



#Initial Gamma Matrix Plotting
num_bins=50
val_1=selected_initial_gm_matrix_data.shape[1]
val_2=selected_initial_gm_matrix_data.shape[2]
fig,ax=plt.subplots(val_1,val_2)
fig.set_figheight(val_1*5)
fig.set_figwidth(val_2*7)
fig.suptitle("Initial Gamma Array")

for a in range(val_1):
    for b in range(val_2):
        if(val_2==1):
            axis=ax
        elif(val_1==1):
            axis=ax[a]
        else:
            axis=ax[a,b]
        try:
            init_data=selected_initial_gm_matrix_data[:,a,b]
            init_std=np.std(init_data)
            if(init_std==0):
                raise ValueError('No standard deviation')
            init_bins=np.linspace(-3*init_std,3*init_std,num_bins)
            selected_data=selected_true_gamma_data[:,a,b]
            selected_std=np.std(selected_data)
            selected_bins=np.linspace(-3*selected_std,3*selected_std,num_bins)
            x=np.linspace(selected_bins[0],selected_bins[-1],200)
            gauss=(1/np.sqrt(selected_std*2*np.pi))*np.exp(-0.5*(np.power(x,2)/np.power(selected_std,2)))

            axis.plot(x,gauss,color="black",label="Gaussian",zorder=-1)
            axis.hist(selected_data,bins=selected_bins,color="black",alpha=0.5,density=True,label="True Values")
            axis.hist(init_data,bins=init_bins,color="purple",alpha=0.75,density=True,label="Initial Values")
            axis.set_ylabel("Relative Density")
            axis.set_xlabel("Gamma Matrix Value")
            axis.text(0.05, 0.93,'Variance: {br:.6f}'.format(br=init_std**2),style='italic',transform=axis.transAxes,va="top",
                      bbox={'facecolor':'lightgrey','alpha':1,'pad':10})
            axis.legend()
        except:
            fig.delaxes(axis)
plt.savefig(output_path+"/Initial Gamma Matrix Values Plot.png")



#Final Gamma Matrix Plotting
num_bins=50
val_1=selected_final_gm_matrix_data.shape[1]
val_2=selected_final_gm_matrix_data.shape[2]
fig,ax=plt.subplots(val_1,val_2)
fig.set_figheight(val_1*5)
fig.set_figwidth(val_2*7)
fig.suptitle("Final Gamma Array")

for a in range(val_1):
    for b in range(val_2):
        if(val_2==1):
            axis=ax
        elif(val_1==1):
            axis=ax[a]
        else:
            axis=ax[a,b]
        try:
            final_data=selected_final_gm_matrix_data[:,a,b]
            final_std=np.std(final_data)
            if(final_std==0):
                raise ValueError('No standard deviation')
            final_bins=np.linspace(-3*final_std,3*final_std,num_bins)
            selected_data=selected_true_gamma_data[:,a,b]
            selected_std=np.std(selected_data)
            selected_bins=np.linspace(-3*selected_std,3*selected_std,num_bins)
            x=np.linspace(selected_bins[0],selected_bins[-1],200)
            gauss=(1/np.sqrt(selected_std*2*np.pi))*np.exp(-0.5*(np.power(x,2)/np.power(selected_std,2)))

            axis.plot(x,gauss,color="black",label="Gaussian",zorder=-1)
            axis.hist(selected_data,bins=selected_bins,color="black",alpha=0.5,density=True,label="True Values")
            axis.hist(final_data,bins=final_bins,color="red",alpha=0.75,density=True,label="Final Values")
            axis.set_ylabel("Relative Density")
            axis.set_xlabel("Gamma Matrix Value")
            axis.text(0.05, 0.93,'Variance: {br:.6f}'.format(br=final_std**2),style='italic',transform=axis.transAxes,va="top",
                      bbox={'facecolor':'lightgrey','alpha':1,'pad':10})
            axis.legend()
        except:
            fig.delaxes(axis)
plt.savefig(output_path+"/Final Gamma Matrix Values Plot.png")

val_1=np.abs(selected_final_gm_matrix_data[:,-1,-1])
val_2=(val_1<0.1).astype(int)
print(val_2[val_2==1].size)

plt.close("all")
plt.plot(val_1)
plt.plot(val_2-1)
plt.show()

# #Final Gamma Matrix Squared Plotting
# num_bins=50
# val_1=selected_final_gm_matrix_data.shape[1]
# val_2=selected_final_gm_matrix_data.shape[2]
# fig,ax=plt.subplots(val_1,val_2)
# fig.set_figheight(val_1*5)
# fig.set_figwidth(val_2*7)
# fig.suptitle("Final Gamma Array Squared")

# for a in range(val_1):
#     for b in range(val_2):
#         if(val_2==1):
#             axis=ax
#         elif(val_1==1):
#             axis=ax[a]
#         else:
#             axis=ax[a,b]
#         final_data=selected_final_gm_matrix_data[:,a,b]
#         final_std=np.std(final_data)
#         if(not(final_std==0)):
#             final_bins=np.linspace(0,3*(final_std**2),num_bins)
#             selected_data=selected_true_gamma_data[:,a,b]
#             selected_std=np.std(selected_data)
#             selected_bins=np.linspace(0,3*(selected_std**2),num_bins)
#             x=np.linspace(selected_bins[0],selected_bins[-1],200)
#             gauss=(1/np.sqrt(selected_std*2*np.pi))*np.exp(-0.5*(np.power(x,2)/np.power(selected_std,2)))

#             axis.plot(x,gauss,color="black",label="Gaussian",zorder=-1)
#             n, bins, rects=axis.hist(np.power(selected_data,2),bins=selected_bins,color="black",alpha=0.5,density=False,label="True Values")
#             max_height_1=0
#             for r in rects:
#                 if(r.get_height()>max_height_1):
#                     max_height_1=r.get_height()
#             n, bins, rects=axis.hist(np.power(final_data,2),bins=final_bins,color="purple",alpha=0.75,density=True,label="Final Values")
#             max_height_2=0
#             for r in rects:
#                 if(r.get_height()>max_height_2):
#                     max_height_2=r.get_height()
#             for r in rects:
#                 r.set_height(r.get_height()*(0.5*max_height_1/max_height_2))
#             ylim=axis.get_ylim()[1]
#             axis.vlines(np.mean(selected_data**2),0,axis.get_ylim()[1],color="Black")
#             axis.set_ylabel("Relative Density",fontsize=16)
#             axis.set_xlabel("Gamma Matrix Value",fontsize=16)
#             axis.set_ylim(0,ylim)
#             axis.set_xlim(0,np.max([np.max(selected_bins),np.max(final_bins)]))
#             axis.legend(prop={'size': 16})
#         else:
#             fig.delaxes(axis)
# plt.savefig(output_path+"/Final Gamma Matrix squared Values Plot.png")



#Final Results Plotting
num_bins=100
fig,ax=plt.subplots(1)
fig.set_figheight(5)
fig.set_figwidth(7)
fig.suptitle("Final Results")

ax.hist(selected_results_data,bins=num_bins,color="green",alpha=0.75,density=True)
axis.set_ylabel("Relative Density")
axis.set_xlabel("Mean Squared Error")
plt.savefig(output_path+"/Final Results.png")



# #Distribution Movement Plotting
# plt.close('all')
# val_1=selected_final_gm_matrix_data.shape[1]
# val_2=selected_final_gm_matrix_data.shape[2]
# fig,axis=plt.subplots()
# fig.subplots_adjust(left=0.25)
# fig.set_figheight(15)
# fig.set_figwidth(20)
# fig.suptitle("Movement of Final Gamma Array Values")

# global row_num
# global col_num
# row_num=0
# col_num=0
# im=axis.scatter(np.arange(0,selected_results_data.size),selected_final_gm_matrix_data[:,row_num,col_num], c=selected_results_data,s=15,zorder=3,cmap=cm.Reds)
# cbar=fig.colorbar(im)
# cbar.set_label("Squared Error of Result")

# def update_plot():
#     try:
#         final_data=selected_final_gm_matrix_data[:,row_num,col_num]
#         final_std=np.std(final_data)
#         if(final_std==0):
#             raise ValueError('No standard deviation')
#         selected_data=selected_true_gamma_data[:,row_num,col_num]
#         indices=np.argsort(selected_data)
#         final_data=final_data[indices]
#         selected_data=selected_data[indices]
#         final_data_error=selected_results_data[indices]
#         x=np.arange(0,indices.size)
#         y=np.linspace(np.min(selected_data),np.max(selected_data),500)
#         gauss=(1/np.sqrt(np.std(selected_data)*2*np.pi))*np.exp(-0.5*(np.power(y,2)/np.power(np.std(selected_data),2)))
#         integral=np.zeros(len(y))
#         for idx in range(len(y)):
#             integral[idx]=np.trapezoid(gauss[:idx],y[:idx])
#         integral*=np.max(x)

#         axis.cla()
#         axis.plot(integral,y,color="grey",label="Gaussian",zorder=0)
#         axis.plot(x,selected_data,color="cyan",alpha=0.75,label="True Values",zorder=2)
#         axis.plot(x,final_data,color="black",alpha=0.75,label="Fitted Values",zorder=2)
#         axis.scatter(x,final_data, c=final_data_error,s=15,zorder=3,cmap=cm.Reds)
#         axis.scatter(x,final_data, c=final_data_error,s=15,zorder=3,cmap=cm.Reds)
        

#         axis.hlines(0,0,np.max(x),colors="black",zorder=-1)
#         axis.fill_between(x,selected_data,final_data,color="black",alpha=0.5,zorder=1)
#         axis.set_ylabel("Gamma Matrix Value")
#         axis.text(0.05, 0.93,'Variance: {br:.6f}'.format(br=final_std**2),style='italic',transform=axis.transAxes,va="top",
#                 bbox={'facecolor':'lightgrey','alpha':1,'pad':10})
#         axis.legend(loc='lower right')
#         axis.set_xlim((0,np.max(x)))
#     except:
#         axis.cla()


# def update_row(input):
#     global row_num
#     row_num=input
#     update_plot()

# def update_col(input):
#     global col_num
#     col_num=input
#     update_plot()

# ax_row = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
# ax_col = fig.add_axes([0.15, 0.25, 0.0225, 0.63])

# row_slider = Slider(
#     ax=ax_row,
#     label="Row",
#     valmin=0,
#     valmax=val_1-1,
#     valinit=row_num,
#     orientation="vertical",
#     valstep=1,
#     color="firebrick"
# )

# col_slider = Slider(
#     ax=ax_col,
#     label="Col",
#     valmin=0,
#     valmax=val_2-1,
#     valinit=col_num,
#     orientation="vertical",
#     valstep=1,
#     color="firebrick"
# )

# ax_row.add_artist(ax_row.yaxis)
# ticks = np.arange(val_1)
# ax_row.set_yticks(ticks)

# ax_col.add_artist(ax_col.yaxis)
# ticks = np.arange(val_2)
# ax_col.set_yticks(ticks)

# row_slider.on_changed(update_row)
# col_slider.on_changed(update_col)

# update_plot()

# plt.show()