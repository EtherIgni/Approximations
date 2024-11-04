from rmatrix import Particle, ElasticChannel, CaptureChannel, SpinGroup
import numpy as np
from copy import deepcopy

class base_reaction():
    def __init__(self) -> None:
        self.capture_channels=[]
        self.incoming_particle=None
        self.outgoing_particle=None
        self.target_particle=None
        self.compound_particle=None
        self.init_guess_full={}
        self.init_guess_diag={}
    def set_incoming(self,incoming_particle:Particle)->None:
        self.incoming_particle=incoming_particle
    def set_outgoing(self,outgoing_particle:Particle)->None:
        self.outgoing_particle=outgoing_particle
    def set_target(self,target_particle:Particle)->None:
        self.target_particle=target_particle
    def set_compound(self,compound_particle:Particle)->None:
        self.compound_particle=compound_particle
    
    def set_elastic_channel(self,J:int,pi:int,ell:int,radius:float,reduced_width_amplitudes:list)->None:
        assert not(self.incoming_particle==None), "No incoming particle defined."
        assert not(self.target_particle==None), "No target particle defined."
        self.elastic_channel=ElasticChannel(self.incoming_particle,
                                            self.target_particle,
                                            J,
                                            pi,
                                            ell,
                                            radius,
                                            reduced_width_amplitudes)
    def get_elastic_channel(self)->ElasticChannel:
        return(self.elastic_channel)
    
    def add_capture_channel(self,J:int,pi:int,ell:int,radius:float,reduced_width_amplitudes:list,excitation:float)->None:
        self.capture_channels.append(CaptureChannel(self.outgoing_particle,self.compound_particle,J,pi,ell,radius,reduced_width_amplitudes, excitation))
    def get_capture_channels(self)->list:
        return(self.capture_channels)
    def remove_capture_channel(self,index:int)->None:
        del self.capture_channels[index]
    def clear_capture_channels(self)->None:
        self.capture_channels=[]
    
    def set_energy_grid(self,energy_grid:np.array)->None:
        self.energy_grid=energy_grid
    def get_energy_grid(self)->np.array:
        return(self.energy_grid)
    
    def set_resonance_energies(self,energies:list)->None:
        self.res_energies=energies
    def get_resonance_energies(self)->list:
        return(self.res_energies)
    
    def establish_spin_group(self)->None:
        self.spin_group=SpinGroup(self.res_energies, self.elastic_channel, self.capture_channels,self.energy_grid)
        self.spin_group.calc_cross_section()
    def get_spin_group(self)->SpinGroup:
        return(self.spin_group)
    
    def get_gamma_matrix(self)->np.array:
        return(np.copy(self.spin_group.gamma_matrix))
    def set_gamma_matrix(self,gamma_matrix:np.array)->None:
        self.spin_group.update_gamma_matrix(gamma_matrix)
    
    def get_L_matrix(self)->np.array:
        return(self.spin_group.L_matrix)
    
    def get_cross_section(self)->np.array:
        return(self.spin_group.total_cross_section)
    
    def get_channels(self)->list:
        return(self.spin_group.channels)
    
    
    
    def get_gm_from_svd(self,svd_matrix):
        U=svd_matrix[:self.get_gamma_matrix().shape[0],0][:,None]
        S=svd_matrix[0,1]
        Vh=svd_matrix[:self.get_gamma_matrix().shape[1],2][None]
        gamma_matrix=(U@Vh)*S
        return(gamma_matrix)
    
    
    
    def evaluate_multi_channel_error_gm(self,gamma_matrix:np.array)->float:
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        total_error=0
        for idc,channel in enumerate(test_spin_group.channels):
            fitted_cross_section=channel.cross_section
            error=np.power((self.spin_group.channels[idc].cross_section-fitted_cross_section),2)
            total_error+=np.sum(error)
        return(total_error)
        
    def evaluate_multi_channel_error_svd(self,svd_matrix:list)->float:
        gamma_matrix=self.get_gm_from_svd(svd_matrix)
        return(self.evaluate_multi_channel_error_gm(gamma_matrix))
    
    def evaluate_total_error_gm(self,gamma_matrix:np.array)->float:
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        error=np.sum(np.power((self.spin_group.total_cross_section-test_spin_group.total_cross_section),2))
        return(error)
    
    def evaluate_real_channel_error_gm(self,gamma_matrix:np.array)->float:
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        
        neutron_error=np.power((self.spin_group.channels[0].cross_section-test_spin_group.channels[0].cross_section),2)
        
        gamma_error=np.zeros(len(neutron_error))
        for idc in range(1,len(test_spin_group.channels)):
            error=self.spin_group.channels[idc].cross_section-test_spin_group.channels[idc].cross_section
            gamma_error+=error
        gamma_error=np.power(gamma_error,2)
        
        total_error=np.sum(neutron_error+gamma_error)
        return(total_error)
    
    def evaluate_total_and_gamma_error_gm(self,gamma_matrix:np.array)->float:
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
    
    
    
    def derivative_of_U_matrix(self,spin_group,gamma_der):
        A_der=-(gamma_der@spin_group.L_matrix@spin_group.gamma_matrix.T+spin_group.gamma_matrix@spin_group.L_matrix@gamma_der.T)
        A_inv_der=-(spin_group.A_matrix@A_der@spin_group.A_matrix)
        W_der=2j*spin_group.P_half@(gamma_der.T@spin_group.A_matrix@spin_group.gamma_matrix+
                     spin_group.gamma_matrix.T@A_inv_der@spin_group.gamma_matrix+
                     spin_group.gamma_matrix.T@spin_group.A_matrix@gamma_der)@spin_group.P_half
        U_der=spin_group.omega_matrix@W_der@spin_group.omega_matrix
        return(U_der)
    
    def derivative_of_elastic_channel(self,spin_group,U_der,channel_num):
        chan_der=10**24 * np.pi/spin_group.k_sq*(-2*U_der[:,0,channel_num].real+np.conjugate(U_der[:,0,channel_num])*spin_group.U_matrix[:,0,channel_num]+np.conjugate(spin_group.U_matrix[:,0,channel_num])*U_der[:,0,channel_num]).real
        return(chan_der)
    
    def derivative_of_capture_channel(self,spin_group,U_der,channel_num):
        chan_der=10**24 * np.pi/spin_group.k_sq *(np.conjugate(U_der[:,0,channel_num])*spin_group.U_matrix[:,0,channel_num]+np.conjugate(spin_group.U_matrix[:,0,channel_num])*U_der[:,0,channel_num]).real
        return(chan_der)
    
    def derivative_multi_channel_error_gm(self,gamma_matrix):
        test_spin_group=deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        gradient=np.zeros(gamma_matrix.shape,float)
        for idr in range(gradient.shape[0]):
            for idc in range(gradient.shape[1]):
                gamma_der=np.zeros(gamma_matrix.shape,float)
                gamma_der[idr,idc]=1
                U_der=self.derivative_of_U_matrix(test_spin_group,gamma_der)
                channel_ders=[]
                for idx,channel in enumerate(test_spin_group.channels):
                    if(isinstance(channel,ElasticChannel)):
                        channel_ders.append(self.derivative_of_elastic_channel(test_spin_group,U_der,idx))
                    elif(isinstance(channel,CaptureChannel)):
                        channel_ders.append(self.derivative_of_capture_channel(test_spin_group,U_der,idx))
                SE_der=0
                for idx in range(len(test_spin_group.channels)):
                    dif=np.sum(-2*(self.spin_group.channels[idx].cross_section-test_spin_group.channels[idx].cross_section)*channel_ders[idx])
                    SE_der+=dif
                gradient[idr,idc]=SE_der
        return(gradient)
    
    def derivative_real_channel_error_gm(self,gamma_matrix):
        test_spin_group=deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        gradient=np.zeros(gamma_matrix.shape,float)
        for idr in range(gradient.shape[0]):
            for idc in range(gradient.shape[1]):
                gamma_der=np.zeros(gamma_matrix.shape,float)
                gamma_der[idr,idc]=1
                U_der=self.derivative_of_U_matrix(test_spin_group,gamma_der)
                channel_ders=[]
                for idx,channel in enumerate(test_spin_group.channels):
                    if(isinstance(channel,ElasticChannel)):
                        channel_ders.append(self.derivative_of_elastic_channel(test_spin_group,U_der,idx))
                    elif(isinstance(channel,CaptureChannel)):
                        channel_ders.append(self.derivative_of_capture_channel(test_spin_group,U_der,idx))
                SE_der=-1*channel_ders[0]*2*(self.spin_group.channels[0].cross_section-test_spin_group.channels[0].cross_section)
                der_set=np.zeros(len(SE_der))
                err_set=np.zeros(len(SE_der))
                for idx in range(1,len(channel_ders)):
                    der_set+=channel_ders[idx]
                    err_set+=self.spin_group.channels[idx].cross_section-test_spin_group.channels[idx].cross_section
                SE_der+=-1*2*der_set*err_set
                gradient[idr,idc]=np.sum(SE_der)
        return(gradient)
    
    def derivative_total_and_gamma_error_gm(self,gamma_matrix):
        test_spin_group=deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        gradient=np.zeros(gamma_matrix.shape,float)
        for idr in range(gradient.shape[0]):
            for idc in range(gradient.shape[1]):
                gamma_der=np.zeros(gamma_matrix.shape,float)
                gamma_der[idr,idc]=1
                U_der=self.derivative_of_U_matrix(test_spin_group,gamma_der)
                channel_ders=[]
                for idx,channel in enumerate(test_spin_group.channels):
                    if(isinstance(channel,ElasticChannel)):
                        channel_ders.append(self.derivative_of_elastic_channel(test_spin_group,U_der,idx))
                    elif(isinstance(channel,CaptureChannel)):
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
                gradient[idr,idc]=np.sum(SE_der)
        return(gradient)
    
    def derivative_total_error_gm(self,gamma_matrix):
        test_spin_group=deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        gradient=np.zeros(gamma_matrix.shape,float)
        for idr in range(gradient.shape[0]):
            for idc in range(gradient.shape[1]):
                gamma_der=np.zeros(gamma_matrix.shape,float)
                gamma_der[idr,idc]=1
                U_der=self.derivative_of_U_matrix(test_spin_group,gamma_der)
                cross_section_der=(10**24*2*np.pi/test_spin_group.k_sq)-(10**24*2*np.pi/test_spin_group.k_sq)*U_der[:,0,0].real
                SE_der=np.sum(-2*(self.spin_group.total_cross_section-test_spin_group.total_cross_section)*cross_section_der)
                gradient[idr,idc]=SE_der
        return(gradient)
    
    def derivative_numeric_gm(self,gamma_matrix,evaluator,step_size_relative):
        result=evaluator(gamma_matrix)
        gradient=np.zeros(gamma_matrix.shape,float)
        for idr in range(gradient.shape[0]):
            for idc in range(gradient.shape[1]):
                gamma_der=np.copy(gamma_matrix)
                gamma_der[idr,idc]+=step_size_relative*gamma_der[idr,idc]
                gradient[idr,idc]=(evaluator(gamma_der)-result)/(step_size_relative*gamma_der[idr,idc])
        return(gradient)
    
    def derivative_numeric_svd(self,svd_matrix):
        step_size_relative=0.0001
        evaluator=self.evaluate_multi_channel_error_svd
        result=evaluator(svd_matrix)
        gradient=np.zeros(svd_matrix.shape,float)
        gamma_shape=self.get_gamma_matrix().shape
        
        for i in range(gamma_shape[0]):
            svd_der=np.copy(svd_matrix)
            step=step_size_relative*svd_der[i,0]
            svd_der[i,0]+=step
            gradient[i,0]=(evaluator(svd_der)-result)/step
        
        svd_der=np.copy(svd_matrix)
        step=step_size_relative*svd_der[0,1]
        svd_der[0,1]+=step
        gradient[0,1]=(evaluator(svd_der)-result)/step
        
        for i in range(gamma_shape[1]):
            svd_der=np.copy(svd_matrix)
            step=step_size_relative*svd_der[i,2]
            svd_der[i,2]+=step
            gradient[i,2]=(evaluator(svd_der)-result)/step
        
        return(gradient)
    
    
    def add_initial_guess_full(self,name:str,guess:np.array)->None:
        self.init_guess_full[name]=guess
        
    def remove_initial_guess_full(self,name:str)->None:
        self.init_guess_full.pop(name)
        
    def clear_initial_guesses_full(self)->None:
        self.init_guess_full={}
        
    def get_initial_guess_full(self,name:str)->np.array:
        return(self.init_guess_full[name])
    def list_initial_geusses_full(self)->list:
        return(self.init_guess_full.keys())