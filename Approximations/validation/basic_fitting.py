import numpy as np
from tools import initial_estimates
from copy import deepcopy
from tools import fitting
import matplotlib.pyplot as plt
from models import basic
from rmatrix import Particle
import numpy as np
import time

def create_leveled_model(binding_gap,excitation_gap,resonance_gap,resonance_distance,energy_grid_buffer):
    model=basic.base_reaction()

    neutron = Particle('n',1,0)
    gamma = Particle('g',0,0)
    target = Particle("20Ne",20,10)
    compound = Particle("20Ne", 20,10, Sn=binding_gap+excitation_gap)
    model.set_incoming(neutron)
    model.set_outgoing(gamma)
    model.set_target(target)
    model.set_compound(compound)

    J = 3
    pi = 1  # positivie parity
    ell = 0  # only s-waves are implemented right now
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [106.78913185, 108.99600881]
    model.set_elastic_channel(J,pi,ell,radius,reduced_width_amplitudes)

    J = 3
    pi = 1  # positive parity
    ell = 1 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [2.51487027e-06, 2.49890268e-06]
    excitation = 0  # the product is left in the ground state 
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    J = 3
    pi = 1  # positive parity
    ell = 2 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [2.51487027e-06*0.8, 2.49890268e-06*0.8]
    excitation = excitation_gap  # the product is left in the 1st ex state at 0.5MeV
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    res_energies=[resonance_distance,resonance_distance+resonance_gap]
    energy_grid=np.linspace(res_energies[0]-energy_grid_buffer,res_energies[1]+energy_grid_buffer,1001)
    model.set_resonance_energies(res_energies)
    model.set_energy_grid(energy_grid)

    model.establish_spin_group()
    
    L_matrix=model.get_L_matrix()
    gamma_matrix=model.get_gamma_matrix()

    level_1_num=(gamma_matrix[0,0]**2)*2*np.mean(L_matrix[:,0,0])
    level_2_num=(gamma_matrix[1,0]**2)*2*np.mean(L_matrix[:,0,0])

    model.clear_capture_channels()
    
    J = 3
    pi = 1  # positive parity
    ell = 1 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [np.real(np.sqrt(level_1_num/(2*np.mean(L_matrix[:,1,1])))),
                                np.real(np.sqrt(level_2_num/(2*np.mean(L_matrix[:,1,1]))))]
    excitation = 0  # the product is left in the ground state 
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    J = 3
    pi = 1  # positive parity
    ell = 2 # orbital ang. momentum of the outgoing primary gamma
    radius = 0.2   # *10^(-12) cm 
    reduced_width_amplitudes = [np.real(np.sqrt(level_1_num/(2*np.mean(L_matrix[:,2,2])))),
                                np.real(np.sqrt(level_2_num/(2*np.mean(L_matrix[:,2,2]))))]
    excitation = excitation_gap  # the product is left in the 1st ex state at 0.5MeV
    model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

    model.establish_spin_group()
    
    return(model)

def run_fit_gamma(problem, problem_name):
    initial_values=problem.get_gamma_matrix()
    initial_values[0,2]=0
    initial_values[1,1]=0
    guess_name="Full Cut"

    iterable=np.array([[1,1,0],[1,0,1]],float)
    gradient_step=float(1000)
    best_fit_matrix=fitting.gradient_descent_half_step(initial_values,
                                                    iterable,
                                                    problem.derivative_real_channel_error_gm,
                                                    gradient_step,
                                                    problem.evaluate_real_channel_error_gm,
                                                    float(1E-6),
                                                    [50,100],
                                                    5,
                                                    0,
                                                    0)
    result=problem.evaluate_multi_channel_error_gm(best_fit_matrix)

    fitted_sg = deepcopy(problem.get_spin_group())
    fitted_sg.update_gamma_matrix(best_fit_matrix)
    initial_sg = deepcopy(problem.get_spin_group())
    initial_sg.update_gamma_matrix(initial_values)

    fig,ax=plt.subplots(2,3)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.suptitle(problem_name+", "+guess_name)

    ax[0,0].clear()
    ax[0,0].plot(problem.get_energy_grid(), problem.get_channels()[0].cross_section, c="b")
    ax[0,0].plot(problem.get_energy_grid(), initial_sg.channels[0].cross_section, c="g", linestyle="dotted")
    ax[0,0].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section, c="r", linestyle="dashed")
    ax[0,0].set_ylabel("Cross Section [b]")
    ax[0,0].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,0].set_title("Neutron Channel")
    ax[0,0].text(((ax[0,0].get_xlim()[1]-ax[0,0].get_xlim()[0])*1/3)+ax[0,0].get_xlim()[0], ((ax[0,0].get_ylim()[1]-ax[0,0].get_ylim()[0])*3/4)+ax[0,0].get_ylim()[0],
                'Best Result: {br:.6f}'.format(br=result), style='italic', bbox={
                'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})

    ax[1,0].clear()
    ax[1,0].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[0].cross_section-fitted_sg.channels[0].cross_section,2)),c="r")
    ax[1,0].set_ylabel("Error in Cross Section")
    ax[1,0].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,0].set_ylim(bottom=0)
    ax[1,0].set_title("Error for Channel")

    ax[0,1].clear()
    ax[0,1].plot(problem.get_energy_grid(), problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section, c="b")
    ax[0,1].plot(problem.get_energy_grid(), fitted_sg.channels[1].cross_section+fitted_sg.channels[2].cross_section, c="r", linestyle="dashed")
    ax[0,1].plot(problem.get_energy_grid(), initial_sg.channels[1].cross_section+initial_sg.channels[2].cross_section, c="g", linestyle="dotted")
    ax[0,1].set_ylabel("Cross Section [b]")
    ax[0,1].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,1].set_title("Gamma Channels ")

    ax[1,1].clear()
    ax[1,1].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section-fitted_sg.channels[1].cross_section-fitted_sg.channels[2].cross_section,2)),c="r")
    ax[1,1].set_ylabel("Error in Cross Section")
    ax[1,1].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,1].set_ylim(bottom=0)
    ax[1,1].set_title("Error for Channel")

    ax[0,2].clear()
    ax[0,2].plot(problem.get_energy_grid(), problem.get_channels()[0].cross_section+problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section, c="b")
    ax[0,2].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section+fitted_sg.channels[1].cross_section+fitted_sg.channels[2].cross_section, c="r", linestyle="dashed")
    ax[0,2].plot(problem.get_energy_grid(), initial_sg.channels[0].cross_section+initial_sg.channels[1].cross_section+initial_sg.channels[2].cross_section, c="g", linestyle="dotted")
    ax[0,2].set_ylabel("Cross Section [b]")
    ax[0,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,2].set_title("Total")

    ax[1,2].clear()
    ax[1,2].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[0].cross_section+problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section-fitted_sg.channels[0].cross_section-fitted_sg.channels[1].cross_section-fitted_sg.channels[2].cross_section,2)),c="r")
    ax[1,2].set_ylabel("Error in Cross Section")
    ax[1,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,2].set_ylim(bottom=0)
    ax[1,2].set_title("Error for Channel")

    plt.savefig("images/real diag runs custom/"+problem_name+", "+guess_name+".png")
    plt.close()

def run_fit_svd(problem, problem_name):
    initial_values=initial_estimates.single_value_approx(problem.get_gamma_matrix())
    guess_name="SVD"

    iterable=np.array([[1,1,1],[1,0,1],[1,0,0]],float)
    gradient_step=float(1000)
    best_fit_matrix=fitting.gradient_descent_half_step(initial_values,
                                                    iterable,
                                                    problem.derivative_numeric_svd,
                                                    gradient_step,
                                                    problem.evaluate_multi_channel_error_svd,
                                                    float(1E-6),
                                                    [50,100],
                                                    5,
                                                    0,
                                                    0)
    result=problem.evaluate_multi_channel_error_svd(best_fit_matrix)

    fitted_sg = deepcopy(problem.get_spin_group())
    fitted_sg.update_gamma_matrix(problem.get_gm_from_svd(best_fit_matrix))
    initial_sg = deepcopy(problem.get_spin_group())
    initial_sg.update_gamma_matrix(problem.get_gm_from_svd(best_fit_matrix))

    fig,ax=plt.subplots(2,3)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.suptitle(problem_name+", "+guess_name)

    ax[0,0].clear()
    ax[0,0].plot(problem.get_energy_grid(), problem.get_channels()[0].cross_section, c="b")
    ax[0,0].plot(problem.get_energy_grid(), initial_sg.channels[0].cross_section, c="g", linestyle="dotted")
    ax[0,0].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section, c="r", linestyle="dashed")
    ax[0,0].set_ylabel("Cross Section [b]")
    ax[0,0].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,0].set_title("Neutron Channel")
    ax[0,0].text(((ax[0,0].get_xlim()[1]-ax[0,0].get_xlim()[0])*1/3)+ax[0,0].get_xlim()[0], ((ax[0,0].get_ylim()[1]-ax[0,0].get_ylim()[0])*3/4)+ax[0,0].get_ylim()[0],
                'Best Result: {br:.6f}'.format(br=result), style='italic', bbox={
                'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})

    ax[1,0].clear()
    ax[1,0].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[0].cross_section-fitted_sg.channels[0].cross_section,2)),c="r")
    ax[1,0].set_ylabel("Error in Cross Section")
    ax[1,0].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,0].set_ylim(bottom=0)
    ax[1,0].set_title("Error for Channel")

    ax[0,1].clear()
    ax[0,1].plot(problem.get_energy_grid(), problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section, c="b")
    ax[0,1].plot(problem.get_energy_grid(), fitted_sg.channels[1].cross_section+fitted_sg.channels[2].cross_section, c="r", linestyle="dashed")
    ax[0,1].plot(problem.get_energy_grid(), initial_sg.channels[1].cross_section+initial_sg.channels[2].cross_section, c="g", linestyle="dotted")
    ax[0,1].set_ylabel("Cross Section [b]")
    ax[0,1].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,1].set_title("Gamma Channels ")

    ax[1,1].clear()
    ax[1,1].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section-fitted_sg.channels[1].cross_section-fitted_sg.channels[2].cross_section,2)),c="r")
    ax[1,1].set_ylabel("Error in Cross Section")
    ax[1,1].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,1].set_ylim(bottom=0)
    ax[1,1].set_title("Error for Channel")

    ax[0,2].clear()
    ax[0,2].plot(problem.get_energy_grid(), problem.get_channels()[0].cross_section+problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section, c="b")
    ax[0,2].plot(problem.get_energy_grid(), fitted_sg.channels[0].cross_section+fitted_sg.channels[1].cross_section+fitted_sg.channels[2].cross_section, c="r", linestyle="dashed")
    ax[0,2].plot(problem.get_energy_grid(), initial_sg.channels[0].cross_section+initial_sg.channels[1].cross_section+initial_sg.channels[2].cross_section, c="g", linestyle="dotted")
    ax[0,2].set_ylabel("Cross Section [b]")
    ax[0,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[0,2].set_title("Total")

    ax[1,2].clear()
    ax[1,2].plot(problem.get_energy_grid(), np.sqrt(np.power(problem.get_channels()[0].cross_section+problem.get_channels()[1].cross_section+problem.get_channels()[2].cross_section-fitted_sg.channels[0].cross_section-fitted_sg.channels[1].cross_section-fitted_sg.channels[2].cross_section,2)),c="r")
    ax[1,2].set_ylabel("Error in Cross Section")
    ax[1,2].set_xlabel("Incident Neutron Energy [eV]")
    ax[1,2].set_ylim(bottom=0)
    ax[1,2].set_title("Error for Channel")

    plt.savefig("images/real svd runs custom/"+problem_name+", "+guess_name+".png")
    plt.close()

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

#energy_brackets=[float(5E2),float(1E3),float(5E3),float(1E4),float(5E4),float(1E5),float(5E5),float(1E6),float(2E3),float(3E6)]
resonance_distances=[float(1E3),float(5E4),float(1E6)]
trials=[]
resonance_distances=[float(1E6)]
trials=[(float(1E5),float(3E6),float(1E5),"1")]

open("runs results.txt", "w").close()

# resonance_gap=float(3E6)
# binding_gap=float(1E5)
# excitation_gap=float(1E5)
# resonance_distance=float(1E5)

#num_runs=len(energy_brackets)**4
num_runs=len(resonance_distances)*len(trials)
current_run=1
printProgressBar(0,num_runs,prefix="Fit Progress:",suffix="Complete",length=50)

for resonance_distance in resonance_distances:
    for energy_values in trials:
        resonance_gap=energy_values[0]
        binding_gap=energy_values[1]
        excitation_gap=energy_values[2]
        
        failed=False
        energy_grid_buffer=0.2*resonance_gap
        suffixes=["","k","M","G"]

        rg_exponent=int(np.floor(np.floor(np.log10(resonance_gap))/3))
        rg_mantisaa=resonance_gap/(10**(np.floor(np.log10(resonance_gap))-np.floor(np.log10(resonance_gap))%3))

        bg_exponent=int(np.floor(np.floor(np.log10(binding_gap))/3))
        bg_mantisaa=binding_gap/(10**(np.floor(np.log10(binding_gap))-np.floor(np.log10(binding_gap))%3))

        eg_exponent=int(np.floor(np.floor(np.log10(excitation_gap))/3))
        eg_mantisaa=excitation_gap/(10**(np.floor(np.log10(excitation_gap))-np.floor(np.log10(excitation_gap))%3))

        rd_exponent=int(np.floor(np.floor(np.log10(resonance_distance))/3))
        rd_mantisaa=resonance_distance/(10**(np.floor(np.log10(resonance_distance))-np.floor(np.log10(resonance_distance))%3))

        problem_name="test"+format(rg_mantisaa,".0f")+" "+suffixes[rg_exponent]+"ev, "+format(bg_mantisaa,".0f")+" "+suffixes[bg_exponent]+"ev, "+format(eg_mantisaa,".0f")+" "+suffixes[eg_exponent]+"ev Bands, "+format(rd_mantisaa,".0f")+" "+suffixes[rd_exponent]+"ev Seperation"
        try:
            problem=create_leveled_model(binding_gap,excitation_gap,resonance_gap,resonance_distance,energy_grid_buffer)
            run_fit_svd(problem,problem_name)
        except:
            failed=True
        with open("runs results.txt", "a") as text_file:
            if(failed):
                addendum=" ✘"
            else:
                addendum=" ✓"
            text_file.write(problem_name+addendum+"\n")
            printProgressBar(current_run,num_runs,prefix="Fit Progress:",suffix="Complete",length=50)
        current_run+=1