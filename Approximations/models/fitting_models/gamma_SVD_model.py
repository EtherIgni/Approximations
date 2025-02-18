import numpy as np
from copy import deepcopy

from Approximations.models import basic

def reshape(gamma_elements:np.array)->np.array:
        num_resonances=int(gamma_elements.size/3)
        num_channels=num_resonances+1

        gamma_matrix=np.zeros((num_resonances,num_channels),float)
        gamma_matrix[:,0]=gamma_elements[:num_resonances]
        U=gamma_elements[num_resonances:num_resonances*2][:,None]
        Vh=gamma_elements[num_resonances*2:num_resonances*3][None,:]
        gamma_matrix[:,1:]=U@Vh
        return(gamma_matrix)

class Gamma_SVD(basic.Fundamental):
    def __init__(self, *args, **kwargs):
        super(Gamma_SVD, self).__init__(*args, **kwargs)
        self.init_guess_full={}
    
    def evaluate(self,gamma_elements:np.array)->float:
        assert not(gamma_elements.size<3*self.num_resonances), "Not Enough gamma elements to derivate"
        assert not(gamma_elements.size>3*self.num_resonances), "Too many gamma elements to derivate"

        gamma_matrix=reshape(gamma_elements)
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        
        errors=np.zeros((self.num_channels,len(self.energy_grid)))
        for idx in range(self.num_channels):
            errors[idx,:]=self.spin_group.channels[idx].cross_section-test_spin_group.channels[idx].cross_section

        total_error=np.sum(np.power(np.sum(errors[1:,:],0),2))+np.sum(np.power(np.sum(errors,0),2))

        return(total_error)
    
    def derivate(self,gamma_elements:np.array)->np.array:
        def evaluator_partial_der(test_spin_group,gamma_der):
                    U_der=self.derivative_of_U_matrix(test_spin_group,gamma_der)

                    channel_ders=np.zeros((self.num_channels,len(self.energy_grid)))
                    channel_ders[0,:]=self.derivative_of_elastic_channel(test_spin_group,U_der)
                    for idx in range(1,len(test_spin_group.channels)):
                        channel_ders[idx,:]=self.derivative_of_capture_channel(test_spin_group,U_der,idx)
                    
                    channel_errors=np.zeros((self.num_channels,len(self.energy_grid)))
                    for idx in range(self.num_channels):
                        channel_errors[idx,:]=self.spin_group.channels[idx].cross_section-test_spin_group.channels[idx].cross_section
                    
                    partial=-np.sum(2*np.sum(channel_errors[1:,:],0)*np.sum(channel_ders[1:,:],0))-np.sum(2*np.sum(channel_errors,0)*np.sum(channel_ders,0))
                    
                    return(partial)
        assert not(gamma_elements.size<3*self.num_resonances), "Not Enough gamma elements to derivate"
        assert not(gamma_elements.size>3*self.num_resonances), "Too many gamma elements to derivate"

        gamma_matrix=reshape(gamma_elements)
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)

        gradient=np.zeros(self.num_resonances*3)

        for idx in range(self.num_resonances):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            gamma_der[idx,0]=1
            gradient[idx]=evaluator_partial_der(test_spin_group,gamma_der)

        for idx in range(self.num_resonances,self.num_resonances*2):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            U=np.zeros(self.num_resonances)
            U[idx%self.num_resonances]=1
            U=U[:,None]
            Vh=gamma_elements[self.num_resonances*2:][None,:]
            gamma_der[:,1:]=U@Vh
            gradient[idx]=evaluator_partial_der(test_spin_group,gamma_der)

        for idx in range(self.num_resonances*2,self.num_resonances*3):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            U=gamma_elements[self.num_resonances:self.num_resonances*2][:,None]
            Vh=np.zeros(self.num_resonances)
            Vh[idx%self.num_resonances]=1
            Vh=Vh[None,:]
            gamma_der[:,1:]=U@Vh
            gradient[idx]=evaluator_partial_der(test_spin_group,gamma_der)

        return(gradient)

    

    def add_initial_guesses(self,name:str,guess:np.array)->None:
        self.init_guess_full[name]=guess
    def remove_initial_guesses(self,name:str)->None:
        self.init_guess_full.pop(name)
    def clear_initial_guesses(self)->None:
        self.init_guess_full={}
    def get_initial_guesses(self,name:str)->np.array:
        return(self.init_guess_full[name])
    def list_initial_guesses(self)->list:
        return(self.init_guess_full.keys())