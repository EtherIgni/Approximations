import numpy as np
from   copy  import deepcopy

from Approximations.models.math_models.R_matrix import RMInterface
from Approximations.rmatrix.base.particles      import Particle






def reshape(gamma_vector:np.array,num_levels:int,num_channels:int)->np.array:
        assert len(gamma_vector.shape) == 1,                 "Reich Moore Evaluation Failed: gamma_vector is not 1 dimensional"
        assert gamma_vector.shape[0]   == num_levels*2,      "Reich Moore Evaluation Failed: gamma_vector has the wrong number of elements"
        gamma_matrix=np.zeros((num_levels,num_channels),float)
        gamma_matrix[:,0]=gamma_vector[:num_levels]
        gamma_matrix[:,1:]=np.diag(gamma_vector[num_levels:])
        return(gamma_matrix)






class ReichMoore():
    def __init__(self,
                 molecular_information,
                 interaction_information,
                 model_information):
        assert "Incident Name"      in molecular_information,   "Model Gen Failed: Incident Name not provided in molecular_information"
        assert "Incident Protons"   in molecular_information,   "Model Gen Failed: Incident Protons not provided in molecular_information"
        assert "Incident Nucleons"  in molecular_information,   "Model Gen Failed: Incident Nucleons not provided in molecular_information"
        assert "Departing Name"     in molecular_information,   "Model Gen Failed: Departing Name not provided in molecular_information"
        assert "Departing Protons"  in molecular_information,   "Model Gen Failed: Departing Protons not provided in molecular_information"
        assert "Departing Nucleons" in molecular_information,   "Model Gen Failed: Departing Nucleons not provided in molecular_information"
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
        energy_grid          = model_information["Energy Grid"]
        self.math_model      = ExtendedRMatrix(self.num_channels,self.num_levels)
    
        #Defines interaction state
        incident  = Particle(molecular_information["Incident Name"],
                             molecular_information["Incident Nucleons"],
                             molecular_information["Incident Protons"])
        departing = Particle(molecular_information["Departing Name"],
                             molecular_information["Departing Nucleons"],
                             molecular_information["Departing Protons"])
        target    = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Nucleons"],
                             molecular_information["Compound Protons"])
        compound  = Particle(molecular_information["Compound Name"],
                             molecular_information["Compound Nucleons"],
                             molecular_information["Compound Protons"],
                             Sn=interaction_information["Separation Energy"])
        self.math_model.set_Incoming(incident)
        self.math_model.set_Outgoing(departing)
        self.math_model.set_Target(target)
        self.math_model.set_Compound(compound)

        #Creates elastic channel
        J                        = 3
        pi                       = 1
        ell                      = 0
        radius                   = 0.2
        reduced_width_amplitudes = np.ones(self.num_levels)
        self.math_model.set_Elastic_Channel(J,
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
            self.math_model.add_Capture_Channel(J,
                                                pi,
                                                ell,
                                                radius,
                                                reduced_width_amplitudes,
                                                excitation)

        #Sets resonances and energy grid
        self.math_model.set_Resonance_Energies(resonances)
        self.math_model.set_Energy_Grid(energy_grid)

        self.math_model.establish_Spin_Group()
    


    def evaluate(self, gamma_vector, data):
        gamma_matrix=reshape(gamma_vector, self.num_levels, self.num_channels)
        return(self.math_model.evaluate(gamma_matrix, data))



    def calc_Gradient(self, gamma_vector, data):
        gamma_matrix=reshape(gamma_vector, self.num_levels, self.num_channels)
        return(self.math_model.calc_Gradient_And_Hessian(gamma_matrix,data,both=False))



    def calc_Gradient_And_Hessian(self, gamma_vector, data):
        gamma_matrix=reshape(gamma_vector, self.num_levels, self.num_channels)
        return(self.math_model.calc_Gradient_And_Hessian(gamma_matrix, data))






class ExtendedRMatrix(RMInterface):
    def __init__(self, *args, **kwargs):
        super(ExtendedRMatrix, self).__init__(*args, **kwargs)
        self.init_guess_full={}



    def evaluate(self,gamma_matrix:np.array,data:np.array)->float:
        test_data = self.get_Data(gamma_matrix)
        
        errors = np.zeros((self.num_channels, len(self.energy_grid)))
        for idx in range(self.num_channels):
            errors[idx,:] = data[1][idx] - test_data[1][idx]

        total_error = np.sum(np.power(np.sum(errors[1:,:], 0), 2) + np.power(np.sum(errors, 0), 2))
        total_error = np.sum(np.power(np.sum(errors[1:,:], 0), 2) + np.power(np.sum(errors, 0), 2))

        return(total_error)



    def calc_Gradient_And_Hessian(self,gamma_matrix,data,both=True):
        data_types      = [float, complex]
        energy_length   = self.energy_grid.size
        num_independent = self.num_levels*2


        #Creating a copy of the spin model to calculate values at the current position
        test_data = self.get_Data(gamma_matrix)


        #Calculating current element values
        L        = self.spin_group.L_matrix
        P        = self.spin_group.P_half
        omega    = self.spin_group.omega_matrix
        A_inv    = self.spin_group.A_matrix
        U_matrix = self.spin_group.U_matrix
        xs_fit   = np.zeros((self.num_channels,energy_length), data_types[0])
        xs_true  = np.zeros((self.num_channels,energy_length), data_types[0])
        for i in range(self.num_channels):
            xs_fit[i]  = test_data[1][i]
            xs_true[i] = data[1][i]
        

        #Defines and sets up the gradients for the elements of the gamma matrix
        gamma_gradient=np.zeros((num_independent,self.num_levels,self.num_channels),data_types[0])
        for i in range(self.num_levels):
            gamma_gradient[i,i,0]=1
        for j in range(self.num_channels-1):
            gamma_gradient[self.num_levels+j,j,j+1]=1
        

        #Does work for the A Element
        A_gradient=np.zeros((num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
        for i in range(num_independent):
            A_gradient[i]=-gamma_gradient[i]@L@gamma_matrix.T-gamma_matrix@L@gamma_gradient[i].T
        
        if(both):
            A_hessian=np.zeros((num_independent,num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
            for i in range(num_independent):
                for j in range(num_independent):
                    A_hessian[i,j]=-gamma_gradient[i]@L@gamma_gradient[j].T-gamma_gradient[j]@L@gamma_gradient[i].T


        #Does work for the A Inverse Element
        A_inv_gradient=np.zeros((num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
        for i in range(num_independent):
            A_inv_gradient[i]=-A_inv@A_gradient[i]@A_inv
        
        if(both):
            A_inv_hessian=np.zeros((num_independent,num_independent,energy_length,self.num_levels,self.num_levels),data_types[1])
            for i in range(num_independent):
                for j in range(num_independent):
                    A_inv_hessian[i,j]=A_inv@A_gradient[i]@A_inv@A_gradient[j]@A_inv-A_inv@A_hessian[i,j]@A_inv+A_inv@A_gradient[j]@A_inv@A_gradient[i]@A_inv


        #Does work for the U Element
        U_gradient=np.zeros((num_independent,energy_length,self.num_channels,self.num_channels),data_types[1])
        for i in range(num_independent):
            U_gradient[i]=2j*omega@P@(gamma_gradient[i].T@A_inv@gamma_matrix+
                                    gamma_matrix.T@A_inv_gradient[i]@gamma_matrix+
                                    gamma_matrix.T@A_inv@gamma_gradient[i])@P@omega
        
        if(both):
            U_hessian=np.zeros((num_independent,num_independent,energy_length,self.num_channels,self.num_channels),data_types[1])
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
        xs_gradient=np.zeros((num_independent,self.num_channels,energy_length),data_types[0])
        for i in range(num_independent):
            xs_gradient[i,0]=(10**24 * np.pi/self.spin_group.k_sq)*(-2*U_gradient[i,:,0,0].real+(np.conjugate(U_gradient[i,:,0,0])*U_matrix[:,0,0]+np.conjugate(U_matrix[:,0,0])*U_gradient[i,:,0,0]).real)
            for j in range(1,self.num_channels):
                xs_gradient[i,j]=(10**24 * np.pi/self.spin_group.k_sq)*(np.conjugate(U_gradient[i,:,0,j])*U_matrix[:,0,j]+np.conjugate(U_matrix[:,0,j])*U_gradient[i,:,0,j]).real
        
        if(both):
            xs_hessian=np.zeros((num_independent,num_independent,self.num_channels,energy_length),data_types[0])
            for i in range(num_independent):
                for j in range(num_independent):
                    xs_hessian[i,j,0]=(10**24 * np.pi/self.spin_group.k_sq)*(-2*U_hessian[i,j,:,0,0].real+
                                                                            (np.conjugate(U_hessian[i,j,:,0,0]) *U_matrix[:,0,0]+
                                                                            np.conjugate(U_gradient[i,:,0,0])  *U_gradient[j,:,0,0]+
                                                                            np.conjugate(U_gradient[j,:,0,0])  *U_gradient[i,:,0,0]+
                                                                            np.conjugate(U_matrix[:,0,0])      *U_hessian[i,j,:,0,0]).real)
                    for k in range(1,self.num_channels):
                        xs_hessian[i,j,k]=(10**24 * np.pi/self.spin_group.k_sq)*(np.conjugate(U_hessian[i,j,:,0,k]) * U_matrix[:,0,k]+
                                                                                np.conjugate(U_gradient[i,:,0,k])  * U_gradient[j,:,0,k]+
                                                                                np.conjugate(U_gradient[j,:,0,k])  * U_gradient[i,:,0,k]+
                                                                                np.conjugate(U_matrix[:,0,k])      * U_hessian[i,j,:,0,k]).real


        #Does work for the Error Term
        error_gradient=np.zeros(num_independent,data_types[0])
        for i in range(num_independent):
            error_gradient[i]=np.sum(2*np.sum(xs_true-xs_fit,0)*np.sum(-xs_gradient[i],0)+2*np.sum(xs_true[1:]-xs_fit[1:],0)*np.sum(-xs_gradient[i,1:],0))
        
        if(both):
            error_hessian=np.zeros((num_independent,num_independent),data_types[0])
            for i in range(num_independent):
                for j in range(num_independent):
                    error_hessian[i,j]=np.sum(2*np.sum(-xs_gradient[j],0)      * np.sum(-xs_gradient[i],0)+
                                            2*np.sum(xs_true-xs_fit,0)         * np.sum(-xs_hessian[i,j],0)+
                                            2*np.sum(-xs_gradient[j,1:],0)     * np.sum(-xs_gradient[i,1:],0)+
                                            2*np.sum(xs_true[1:]-xs_fit[1:],0) * np.sum(-xs_hessian[i,j,1:],0))
        
        if(both):
            return(error_gradient, error_hessian)
        else:
            return(error_gradient)