from rmatrix import Particle, ElasticChannel, CaptureChannel, SpinGroup
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tools import fitting
import matplotlib.animation as ani
from tools import initial_estimates
from models import neon_bands_3MeV_c2

global running_gamma
global settings
global problem
global results
global results_diff
global ax

problem=neon_bands_3MeV_c2.model
problem_name="3MeV Case 2"



# running_gamma=problem.get_initial_guess_full("Best Guess 1")
# guess_name="Hand Fit"

running_gamma=initial_estimates.diag_approx_guess(problem.get_gamma_matrix(),problem.get_L_matrix())
guess_name="Weighted Average Approx"

# running_gamma=problem.get_gamma_matrix()
# running_gamma[0,2]=0
# running_gamma[1,1]=0
# guess_name="Cut Full Matrix"



num_frames=50

results=[]
results_diff=[]
iterable=np.array([[1,1,1],[1,1,1]],float)
gradient_step=float(1000)
initial_sg = deepcopy(problem.get_spin_group())
initial_sg.update_gamma_matrix(running_gamma)
settings=[1000,0,0]

def step_descent(type):
    global running_gamma
    global settings
    global problem
    if(type=="full"):
        iterable=np.array([[1,1,1],[1,1,1]],float)
    elif(type=="diag"):
        iterable=np.array([[1,1,0],[1,0,1]],float)
    gradient_step=float(1000)
    running_gamma, settings=fitting.gradient_descent_half_step(running_gamma,
                                                    iterable,
                                                    problem.derivative_of_SqrErr_channel_gm,
                                                    gradient_step,
                                                    problem.evaluate_multi_channel_error_gm,
                                                    float(1E-6),
                                                    [1,100],
                                                    5,
                                                    0,
                                                    previous_settings=settings)
    

def update(frame):
    global ax
    global results
    global results_diff
    fitted_sg = deepcopy(problem.get_spin_group())
    fitted_sg.update_gamma_matrix(running_gamma)
    results.append(np.sqrt(problem.evaluate_multi_channel_error_gm(running_gamma)))
    if(len(results)>1):
        results_diff.append(results[-1]-results[-2])
    else:
        results_diff.append(0)
    
    ax[0,0].clear()
    ax[0,0].plot(problem.get_energy_grid(), problem.get_channels()[0].cross_section, c="b")
    ax[0,0].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section, c="r", linestyle="dashed")
    ax[0,0].plot(problem.get_energy_grid(), initial_sg.channels[0].cross_section, c="g", linestyle="dotted")
    ax[0,0].set_ylabel("Cross Section [b]")
    ax[0,0].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,0].set_title("Channel 0")

    ax[1,0].clear()
    ax[1,0].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[0].cross_section-fitted_sg.channels[0].cross_section,2)),c="r")
    ax[1,0].set_ylabel("Error in Cross Section")
    ax[1,0].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,0].set_ylim(bottom=0)
    ax[1,0].set_title("Error for Channel")

    ax[0,1].clear()
    ax[0,1].plot(problem.get_energy_grid(), problem.get_channels()[1].cross_section, c="b")
    ax[0,1].plot(problem.get_energy_grid(), fitted_sg.channels[1].cross_section, c="r", linestyle="dashed")
    ax[0,1].plot(problem.get_energy_grid(), initial_sg.channels[1].cross_section, c="g", linestyle="dotted")
    ax[0,1].set_ylabel("Cross Section [b]")
    ax[0,1].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,1].set_title("Channel 1")

    ax[1,1].clear()
    ax[1,1].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[1].cross_section-fitted_sg.channels[1].cross_section,2)),c="r")
    ax[1,1].set_ylabel("Error in Cross Section")
    ax[1,1].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,1].set_ylim(bottom=0)
    ax[1,1].set_title("Error for Channel")

    ax[0,2].clear()
    ax[0,2].plot(problem.get_energy_grid(), problem.get_channels()[2].cross_section, c="b")
    ax[0,2].plot(problem.get_energy_grid(), fitted_sg.channels[2].cross_section, c="r", linestyle="dashed")
    ax[0,2].plot(problem.get_energy_grid(), initial_sg.channels[2].cross_section, c="g", linestyle="dotted")
    ax[0,2].set_ylabel("Cross Section [b]")
    ax[0,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,2].set_title("Channel 2")

    ax[1,2].clear()
    ax[1,2].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[2].cross_section-fitted_sg.channels[2].cross_section,2)),c="r")
    ax[1,2].set_ylabel("Error in Cross Section")
    ax[1,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,2].set_ylim(bottom=0)
    ax[1,2].set_title("Error for Channel")
    
    ax[0,3].clear()
    ax[0,3].plot(np.arange(1,len(results)+1,1),results,c="k")
    ax[0,3].set_xlim(1,num_frames)
    ax[0,3].set_ylim(bottom=0)
    ax[0,3].set_ylabel("Total Channel Error")
    ax[0,3].set_xlabel("Iteration")
    
    ax[1,3].clear()
    ax[1,3].plot(np.arange(1,len(results_diff)+1,1),results_diff,c="k")
    ax[1,3].set_xlim(1,num_frames)
    ax[1,3].set_ylim(top=0)
    ax[1,3].set_ylabel("Change in Total Channel Error")
    ax[1,3].set_xlabel("Iteration")

fig,ax=plt.subplots(2,4)
fig.set_figheight(10)
fig.set_figwidth(28)
animation=ani.FuncAnimation(fig,update,frames=num_frames,repeat=True,interval=40)
animation.save("gifs/"+problem_name+", "+guess_name+".gif",writer=ani.PillowWriter(fps=25))