import numpy as np
from copy import deepcopy
from models import basic

class Reich_Moore(basic.Fundamental):
    def __init__(self, *args, **kwargs):
        super(Reich_Moore, self).__init__(*args, **kwargs)
        self.init_guess_full={}
    


    def reshape(self,gamma_elements:np.array)->np.array:
        assert not(gamma_elements.size<2*self.num_resonances), "Not Enough gamma elements to derivate"
        assert not(gamma_elements.size>2*self.num_resonances), "Too many gamma elements to derivate"
        gamma_matrix=np.zeros((self.num_resonances,self.num_channels),float)
        gamma_matrix[:,0]=gamma_elements[:self.num_resonances]
        gamma_matrix[:,1:]=np.diag(gamma_elements[self.num_resonances:self.num_resonances*2])
        return(gamma_matrix)
    
    def evaluate(self,gamma_elements:np.array)->float:
        test_spin_group = deepcopy(self.spin_group)
        gamma_matrix=self.reshape(gamma_elements)
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