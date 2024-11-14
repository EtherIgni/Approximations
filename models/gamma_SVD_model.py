import numpy as np
from models import basic
from copy import deepcopy

class Gamma_SVD(basic.Fundamental):
    def __init__(self, *args, **kwargs):
        super(Gamma_SVD, self).__init__(*args, **kwargs)
        self.init_guess_full={}
    


    def reshape(self,gamma_elements:np.array)->np.array:
        assert not(gamma_elements.size<3*self.num_resonances), "Not Enough gamma elements to derivate"
        assert not(gamma_elements.size>3*self.num_resonances), "Too many gamma elements to derivate"
        gamma_matrix=np.zeros((self.num_resonances,self.num_channels),float)
        gamma_matrix[:,0]=gamma_elements[:self.num_resonances]
        U=gamma_elements[self.num_resonances:self.num_resonances*2][:,None]
        Vh=gamma_elements[self.num_resonances*2:self.num_resonances*3]
        gamma_matrix[:,1:]=U@Vh
        return(gamma_matrix)
    
    def evaluate(self,gamma_elements:np.array)->float:
        gamma_matrix=self.reshape(gamma_elements)
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        
        total_error=np.zeros(len(self.energy_grid))
        
        partial_error=np.zeros(len(total_error))
        for idc in range(0,len(test_spin_group.channels)):
            error=self.spin_group.channels[idc].cross_section-test_spin_group.channels[idc].cross_section
            partial_error+=error
        partial_error=np.power(partial_error,2)
        total_error+=partial_error
        
        partial_error=np.zeros(len(total_error))
        for idc in range(1,len(test_spin_group.channels)):
            error=self.spin_group.channels[idc].cross_section-test_spin_group.channels[idc].cross_section
            partial_error+=error
        partial_error=np.power(partial_error,2)
        total_error+=partial_error
        
        total_error=np.sum(total_error)
        return(total_error)
    
    def derivate(self,gamma_elements:np.array)->np.array:
        def evaluator_derivative(test_spin_group,gamma_der):
                    U_der=self.derivative_of_U_matrix(test_spin_group,gamma_der)
                    channel_ders=[self.derivative_of_elastic_channel(test_spin_group,U_der,idx)]
                    for idx in range(1,len(test_spin_group.channels)):
                        channel_ders.append(self.derivative_of_capture_channel(test_spin_group,U_der,idx))
                    SE_der=np.zeros(len(self.energy_grid))
                    der_set=np.zeros(len(SE_der))
                    err_set=np.zeros(len(SE_der))
                    for idx in range(0,len(channel_ders)):
                        der_set+=channel_ders[idx]
                        err_set+=self.spin_group.channels[idx].cross_section-test_spin_group.channels[idx].cross_section
                    SE_der+=-1*2*der_set*err_set
                    der_set=np.zeros(len(SE_der))
                    err_set=np.zeros(len(SE_der))
                    for idx in range(1,len(channel_ders)):
                        der_set+=channel_ders[idx]
                        err_set+=self.spin_group.channels[idx].cross_section-test_spin_group.channels[idx].cross_section
                    SE_der+=-1*2*der_set*err_set
                    return(np.sum(SE_der))
        gamma_matrix=self.reshape(gamma_elements)
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)

        derivative=np.zeros(self.num_resonances*3)
        for idx in range(self.num_resonances):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            gamma_der[idx,0]=1
            derivative[idx]=evaluator_derivative(test_spin_group,gamma_der)
        for idx in range(self.num_resonances,self.num_resonances*2):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            U=np.zeros(self.num_resonances)
            U[idx]=1
            U=U[:,None%self.num_resonances]
            Vh=gamma_elements[self.num_resonances*2:]
            gamma_der[:,1:]=U@Vh
            derivative[idx]=evaluator_derivative(test_spin_group,gamma_der)
        for idx in range(self.num_resonances*2,self.num_resonances*3):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            U=gamma_elements[self.num_resonances:self.num_resonances*2][:,None]
            Vh=np.zeros(self.num_resonances)
            Vh[idx%self.num_resonances]=1
            gamma_der[:,1:]=U@Vh
            derivative[idx]=evaluator_derivative(test_spin_group,gamma_der)
        return(derivative)




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