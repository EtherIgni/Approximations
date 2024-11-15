import numpy as np
from copy import deepcopy

from Approximations.models import basic

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
        gamma_matrix=self.reshape(gamma_elements)
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
                    
                    partial=np.sum(-2*np.sum(channel_errors[1:,:],0)*np.sum(channel_errors[1:,:],0)-2*np.sum(channel_errors,0)*np.sum(channel_errors,0))
                    
                    return(partial)
        
        gamma_matrix=self.reshape(gamma_elements)
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)

        gradient=np.zeros(self.num_resonances*3)

        for idx in range(self.num_resonances):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            gamma_der[idx,0]=1
            gradient[idx]=evaluator_partial_der(test_spin_group,gamma_der)

        for idx in range(self.num_resonances,self.num_resonances*2):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            gamma_der[idx%self.num_resonances,1+(idx%self.num_resonances)]=1
            gradient[idx]=evaluator_partial_der(test_spin_group,gamma_der)

        return(gradient)
    


    def calc_hessian_and_gradient(self,gamma_matrix,iterable_mapping):
        data_types=[float,complex]
        energy_length=self.energy_grid.size
        gamma_shape=gamma_matrix.shape
        num_independent=iterable_mapping.shape[0]
        
        test_spin_group=deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        L=self.spin_group.L_matrix
        P=self.spin_group.P_half
        omega=self.spin_group.omega_matrix
        A_inv=self.spin_group.A_matrix
        U_matrix=self.spin_group.U_matrix
        xs_fit=np.zeros((gamma_shape[1],energy_length),data_types[0])
        for i in range(gamma_shape[1]):
            xs_fit[i]=test_spin_group.channels[i].cross_section
        xs_true=np.zeros((gamma_shape[1],energy_length),data_types[0])
        for i in range(gamma_shape[1]):
            xs_true[i]=self.spin_group.channels[i].cross_section
        
        gamma_gradient=np.zeros((6,2,3),data_types[0])
        for i in range(num_independent):
            gamma_gradient[i,iterable_mapping[i,0],iterable_mapping[i,1]]=1
        
        
        A_gradient=np.zeros((num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_types[1])
        for i in range(num_independent):
            A_gradient[i]=-gamma_gradient[i]@L@gamma_matrix.T-gamma_matrix@L@gamma_gradient[i].T
        
        A_hessian=np.zeros((num_independent,num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_types[1])
        for i in range(num_independent):
            for j in range(num_independent):
                if(iterable_mapping[i,1]==iterable_mapping[j,1]):
                    A_hessian[i,j]=-gamma_gradient[i]@L@gamma_gradient[j].T-gamma_gradient[j]@L@gamma_gradient[i].T
                else:
                    A_hessian[i,j]=0


        A_inv_gradient=np.zeros((num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_types[1])
        for i in range(num_independent):
            A_inv_gradient[i]=-A_inv@A_gradient[i]@A_inv
        
        A_inv_hessian=np.zeros((num_independent,num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_types[1])
        for i in range(num_independent):
            for j in range(num_independent):
                A_inv_hessian[i,j]=A_inv@A_gradient[i]@A_inv@A_gradient[j]@A_inv-A_inv@A_hessian[i,j]@A_inv+A_inv@A_gradient[j]@A_inv@A_gradient[i]@A_inv


        U_gradient=np.zeros((num_independent,energy_length,gamma_shape[1],gamma_shape[1]),data_types[1])
        for i in range(num_independent):
            U_gradient[i]=2j*omega@P@(gamma_gradient[i].T@A_inv@gamma_matrix+
                                    gamma_matrix.T@A_inv_gradient[i]@gamma_matrix+
                                    gamma_matrix.T@A_inv@gamma_gradient[i])@P@omega
        U_hessian=np.zeros((num_independent,num_independent,energy_length,gamma_shape[1],gamma_shape[1]),data_types[1])
        for i in range(num_independent):
            for j in range(num_independent):
                interstage =gamma_gradient[i].T@A_inv_gradient[j]@gamma_matrix
                interstage+=gamma_gradient[i].T@A_inv@gamma_gradient[j]
                interstage+=gamma_gradient[j].T@A_inv_gradient[i]@gamma_matrix
                interstage+=gamma_matrix.T@A_inv_hessian[i,j]@gamma_matrix
                interstage+=gamma_matrix.T@A_inv_gradient[i]@gamma_gradient[j]
                interstage+=gamma_gradient[j].T@A_inv@gamma_gradient[i]
                interstage+=gamma_matrix.T@A_inv_gradient[j]@gamma_gradient[i]
                U_hessian[i,j]=2j*omega@P@interstage@P@omega


        xs_gradient=np.zeros((num_independent,gamma_shape[1],energy_length),data_types[0])
        for i in range(num_independent):
            xs_gradient[i,0]=(10**24 * np.pi/self.spin_group.k_sq)*(-2*U_gradient[i,:,0,0].real+(np.conjugate(U_gradient[i,:,0,0])*U_matrix[:,0,0]+np.conjugate(U_matrix[:,0,0])*U_gradient[i,:,0,0]).real)
            for j in range(1,gamma_shape[1]):
                xs_gradient[i,j]=(10**24 * np.pi/self.spin_group.k_sq)*(np.conjugate(U_gradient[i,:,0,j])*U_matrix[:,0,j]+np.conjugate(U_matrix[:,0,j])*U_gradient[i,:,0,j]).real
                
        xs_hessian=np.zeros((num_independent,num_independent,gamma_shape[1],energy_length),data_types[0])
        for i in range(num_independent):
            for j in range(num_independent):
                xs_hessian[i,j,0]=(10**24 * np.pi/self.spin_group.k_sq)*(-2*U_hessian[i,j,:,0,0].real+
                                                                         (np.conjugate(U_hessian[i,j,:,0,0]) *U_matrix[:,0,0]+
                                                                         np.conjugate(U_gradient[i,:,0,0])  *U_gradient[j,:,0,0]+
                                                                         np.conjugate(U_gradient[j,:,0,0])  *U_gradient[i,:,0,0]+
                                                                         np.conjugate(U_matrix[:,0,0])      *U_hessian[i,j,:,0,0]).real)
                for k in range(1,gamma_shape[1]):
                    xs_hessian[i,j,k]=(10**24 * np.pi/self.spin_group.k_sq)*(np.conjugate(U_hessian[i,j,:,0,k]) * U_matrix[:,0,k]+
                                                                             np.conjugate(U_gradient[i,:,0,k])  * U_gradient[j,:,0,k]+
                                                                             np.conjugate(U_gradient[j,:,0,k])  * U_gradient[i,:,0,k]+
                                                                             np.conjugate(U_matrix[:,0,k])      * U_hessian[i,j,:,0,k]).real


        error_gradient=np.zeros(num_independent,data_types[0])
        for i in range(num_independent):
            error_gradient[i]=np.sum(2*np.sum(xs_true-xs_fit,0)*np.sum(-xs_gradient[i],0)+2*np.sum(xs_true[1:]-xs_fit[1:],0)*np.sum(-xs_gradient[i,1:],0))
            
        error_hessian=np.zeros((num_independent,num_independent),data_types[0])
        for i in range(num_independent):
            for j in range(num_independent):
                error_hessian[i,j]=np.sum(2*np.sum(-xs_gradient[j],0)      * np.sum(-xs_gradient[i],0)+
                                        2*np.sum(xs_true-xs_fit,0)         * np.sum(-xs_hessian[i,j],0)+
                                        2*np.sum(-xs_gradient[j,1:],0)     * np.sum(-xs_gradient[i,1:],0)+
                                        2*np.sum(xs_true[1:]-xs_fit[1:],0) * np.sum(-xs_hessian[i,j,1:],0))
        
        return(error_gradient,error_hessian)
    
    

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