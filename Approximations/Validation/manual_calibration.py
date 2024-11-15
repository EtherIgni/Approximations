from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from models import neon_bands_3MeV_c4

global problem
global ax
global running_gamma

problem=neon_bands_3MeV_c4.model
problem_name="Case 4, 0.5Mev to 3MeV"

running_gamma=problem.get_gamma_matrix()
running_gamma[0,2]=0
running_gamma[1,1]=0

print(running_gamma)

def update_graph():
    global problem
    global ax
    global running_gamma
    fitted_sg = deepcopy(problem.get_spin_group())
    fitted_sg.update_gamma_matrix(running_gamma)
    
    ax[0,0].clear()
    ax[0,0].plot(problem.get_energy_grid(), problem.get_channels()[0].cross_section, c="b")
    ax[0,0].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section, c="r", linestyle="dashed")
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
    ax[0,2].set_ylabel("Cross Section [b]")
    ax[0,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,2].set_title("Channel 2")

    ax[1,2].clear()
    ax[1,2].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[2].cross_section-fitted_sg.channels[2].cross_section,2)),c="r")
    ax[1,2].set_ylabel("Error in Cross Section")
    ax[1,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,2].set_ylim(bottom=0)
    ax[1,2].set_title("Error for Channel")

fig,ax=plt.subplots(2,3)
fig.set_figheight(10)
fig.set_figwidth(22)
fig.subplots_adjust(left=0.3)

gamma_ax_00 = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
gamma_00 = Slider(
    ax=gamma_ax_00,
    label="Gamma 0 0",
    valmin=0,
    valmax=problem.get_gamma_matrix()[0,0]*2,
    valinit=problem.get_gamma_matrix()[0,0],
    orientation="vertical")
def update_gamma_00(value):
    running_gamma[0,0]=value
    update_graph()
gamma_00.on_changed(update_gamma_00)

gamma_ax_01 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
gamma_01 = Slider(
    ax=gamma_ax_01,
    label="Gamma 0 1",
    valmin=0,
    valmax=problem.get_gamma_matrix()[0,1]*2,
    valinit=problem.get_gamma_matrix()[0,1],
    orientation="vertical")
def update_gamma_01(value):
    running_gamma[0,1]=value
    update_graph()
gamma_01.on_changed(update_gamma_01)

gamma_ax_10 = fig.add_axes([0.15, 0.25, 0.0225, 0.63])
gamma_10 = Slider(
    ax=gamma_ax_10,
    label="Gamma 1 0",
    valmin=0,
    valmax=problem.get_gamma_matrix()[1,0]*2,
    valinit=problem.get_gamma_matrix()[1,0],
    orientation="vertical")
def update_gamma_10(value):
    running_gamma[1,0]=value
    update_graph()
gamma_10.on_changed(update_gamma_10)

gamma_ax_12 = fig.add_axes([0.2, 0.25, 0.0225, 0.63])
gamma_12 = Slider(
    ax=gamma_ax_12,
    label="Gamma 1 2",
    valmin=0,
    valmax=problem.get_gamma_matrix()[1,2]*2,
    valinit=problem.get_gamma_matrix()[1,2],
    orientation="vertical")
def update_gamma_12(value):
    running_gamma[1,2]=value
    update_graph()
gamma_12.on_changed(update_gamma_12)

update_graph()
plt.show()

print(running_gamma)