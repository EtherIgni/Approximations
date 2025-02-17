import numpy as np
from copy import deepcopy

from Approximations.models.math_models.R_matrix import RM_Interface






def reshape(gamma_elements:np.array,num_levels:int,num_channels:int)->np.array:
        gamma_matrix=np.zeros((num_levels,num_channels),float)
        gamma_matrix[:,0]=gamma_elements[:num_levels]
        gamma_matrix[:,1:]=np.diag(gamma_elements[num_levels:])
        return(gamma_matrix)






class Reich_Moore():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Incident Name"      in molecular_information,   "Model Gen Failed: Incident Name not provided in molecular_information"
        assert "Incident Protons"   in molecular_information,   "Model Gen Failed: Incident Protons not provided in molecular_information"
        assert "Incident Nucleons"  in molecular_information,   "Model Gen Failed: Incident Nucleons not provided in molecular_information"
        assert "departing Name"     in molecular_information,   "Model Gen Failed: departing Name not provided in molecular_information"
        assert "departing Protons"  in molecular_information,   "Model Gen Failed: departing Protons not provided in molecular_information"
        assert "departing Nucleons" in molecular_information,   "Model Gen Failed: departing Nucleons not provided in molecular_information"
        assert "Compound Name"      in molecular_information,   "Model Gen Failed: Compound Name not provided in molecular_information"
        assert "Compound Protons"   in molecular_information,   "Model Gen Failed: Compound Protons not provided in molecular_information"
        assert "Compound Nucleons"  in molecular_information,   "Model Gen Failed: Compound Nucleons not provided in molecular_information"
        assert "Separation Energy"  in interaction_information, "Model Gen Failed: Separation Energy not provided in interaction_information"
        assert "Number Levels"      in interaction_information, "Model Gen Failed: Number Levels not provided in interaction_information"
        assert "Excited States"     in interaction_information, "Model Gen Failed: Excited States not provided in interaction_information"
        assert "Resonance Levels"   in interaction_information, "Model Gen Failed: Resonance Levels not provided in interaction_information"
        assert "Energy Grid"        in model_information,       "Model Gen Failed: Energy Grid not provided in model_information"
        
        self.init_guess_full = {}
        self.num_levels      = interaction_information["Number Levels"]
        self.num_channels    = self.num_levels+1
        excited_states       = interaction_information["Excited States"][:self.num_levels]
        resonances           = interaction_information["Resonance Levels"]
        energy_grid          = interaction_information["Energy Grid"]
        self.math_model      = RM_Interface(num_channels,self.num_levels)
    
        #Defines interaction state
        incident  = Particle(molecular_information["Incident Name"],
                             molecular_information["Incident Protons"],
                             molecular_information["Incident Nucleons"])
        departing = Particle(molecular_information["departing Name"],
                             molecular_information["departing Protons"],
                             molecular_information["departing Nucleons"])
        target    = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Protons"],
                             molecular_information["Compound Nucleons"])
        compound  = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Protons"],
                             molecular_information["Compound Nucleons"],
                             Sn=interaction_information["Separation Energy"])
        self.math_model.set_incoming(incident)
        self.math_model.set_outgoing(departing)
        self.math_model.set_target(target)
        self.math_model.set_compound(compound)

        #Creates elastic channel
        J                        = 3
        pi                       = 1
        ell                      = 0
        radius                   = 0.2
        reduced_width_amplitudes = np.ones(self.num_levels)
        self.math_model.set_elastic_channel(J,
                                            pi,
                                            ell,
                                            radius,
                                            reduced_width_amplitudes)

        #Creates inelastic channels
        for idx,excitation in enumerate(excited_states):
            J                             = 3
            pi                            = 1
            ell                           = 0
            radius                        = 0.2
            reduced_width_amplitudes      = np.zeros(self.num_levels)
            reduced_width_amplitudes[idx] = 1
            self.math_model.add_capture_channel(J,
                                                pi,
                                                ell,
                                                radius,
                                                reduced_width_amplitudes,
                                                excitation)

        #Sets resonances and energy grid
        self.math_model.set_resonance_energies(resonances)
        self.math_model.set_energy_grid(energy_grid)

        self.math_model.establish_spin_group()
    


    def evaluate(self,gamma_elements:np.array)->float:
        assert not(gamma_elements.size<2*self.num_resonances), "Not Enough gamma elements to derivate"
        assert not(gamma_elements.size>2*self.num_resonances), "Too many gamma elements to derivate"

        gamma_matrix=reshape(gamma_elements)
        test_spin_group = deepcopy(self.math_model.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)
        
        errors=np.zeros((self.num_channels,len(self.energy_grid)))
        for idx in range(self.num_channels):
            errors[idx,:]=self.spin_group.channels[idx].cross_section-test_spin_group.channels[idx].cross_section

        total_error=np.sum(np.power(np.sum(errors[1:,:],0),2)+np.power(np.sum(errors,0),2))
        total_error=np.sum(np.power(np.sum(errors[1:,:],0),2)+np.power(np.sum(errors,0),2))

        return(total_error)
    


    def calcGradient(self,gamma_elements:np.array)->np.array:
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
        assert not(gamma_elements.size<2*self.num_resonances), "Not Enough gamma elements to derivate"
        assert not(gamma_elements.size>2*self.num_resonances), "Too many gamma elements to derivate"

        gamma_matrix=reshape(gamma_elements)
        test_spin_group = deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)

        gradient=np.zeros(self.num_resonances*2)

        for idx in range(self.num_resonances):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            gamma_der[idx,0]=1
            gradient[idx]=evaluator_partial_der(test_spin_group,gamma_der)

        for idx in range(self.num_resonances,self.num_resonances*2):
            gamma_der=np.zeros((self.num_resonances,self.num_channels),float)
            gamma_der[idx%self.num_resonances,1+(idx%self.num_resonances)]=1
            gradient[idx]=evaluator_partial_der(test_spin_group,gamma_der)

        return(gradient)
    


    def calc_hessian_and_gradient(self,gamma_vector):
        data_types=[float,complex]
        energy_length=self.energy_grid.size
        gamma_shape=self.get_gamma_matrix().shape
        num_levels=gamma_shape[0]
        num_channels=gamma_shape[1]
        gamma_matrix=reshape(gamma_vector)
        num_independent=len(gamma_vector)


        #Creating a copy of the spin model to calculate values at the current position
        test_spin_group=deepcopy(self.spin_group)
        test_spin_group.update_gamma_matrix(gamma_matrix)

        #Calculating current element values
        L=self.spin_group.L_matrix
        P=self.spin_group.P_half
        omega=self.spin_group.omega_matrix
        A_inv=self.spin_group.A_matrix
        U_matrix=self.spin_group.U_matrix
        xs_fit=np.zeros((num_channels,energy_length),data_types[0])
        for i in range(num_channels):
            xs_fit[i]=test_spin_group.channels[i].cross_section
        xs_true=np.zeros((num_channels,energy_length),data_types[0])
        for i in range(num_channels):
            xs_true[i]=self.spin_group.channels[i].cross_section
        
        #Defines and sets up the gradients for the elements of the gamma matrix
        gamma_gradient=np.zeros((num_independent,num_levels,num_channels),data_types[0])
        for i in range(num_levels):
            gamma_gradient[i,i,0]=1
        for j in range(num_channels-1):
            gamma_gradient[num_levels+j,j,j+1]=1
        

        #Does work for the A Element
        A_gradient=np.zeros((num_independent,energy_length,num_levels,num_levels),data_types[1])
        for i in range(num_independent):
            A_gradient[i]=-gamma_gradient[i]@L@gamma_matrix.T-gamma_matrix@L@gamma_gradient[i].T
        
        A_hessian=np.zeros((num_independent,num_independent,energy_length,num_levels,num_levels),data_types[1])
        for i in range(num_independent):
            for j in range(num_independent):
                A_hessian[i,j]=-gamma_gradient[i]@L@gamma_gradient[j].T-gamma_gradient[j]@L@gamma_gradient[i].T


        #Does work for the A Inverse Element
        A_inv_gradient=np.zeros((num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_types[1])
        for i in range(num_independent):
            A_inv_gradient[i]=-A_inv@A_gradient[i]@A_inv
        
        A_inv_hessian=np.zeros((num_independent,num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_types[1])
        for i in range(num_independent):
            for j in range(num_independent):
                A_inv_hessian[i,j]=A_inv@A_gradient[i]@A_inv@A_gradient[j]@A_inv-A_inv@A_hessian[i,j]@A_inv+A_inv@A_gradient[j]@A_inv@A_gradient[i]@A_inv


        #Does work for the U Element
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


        #Does work for the cross section
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


        #Does work for the Error Term
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