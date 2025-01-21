import numpy as np 

class SpinGroup:
    def __init__(self, res_energies, incident_channel, outgoing_channels, energy_grid, debug=False):
        """ Class to hold a single spin group
        
        Parameters
        ----------
        res_energies : list or numpy array
            The resonances for this spin group, in eV

        incident_channel : ElasticChannel obj
            The incident channel

        outgoing_channels : ElasticChannel or CaptureChannel obj or list of objs
            The outgoing channel(s) for the spin group

        energy_grid : list or numpy array
            The incident neutron energy grid to calculate the cross section on, in eV

        debug : bool, optional, default is False
            Debug mode prints out most of the matrices for the first incident energy 
        
        
        Attributes
        ----------
        res_energies : numpy array
            The resonances for this spin group, in eV

        incident_channel : ElasticChannel obj
            The incident channel

        channels :  list of ElasticChannel / CaptureChannel objs
            All of the channels for the spin group, starting with the incident channel

        energy_grid : numpy array
            The incident neutron energy grid to calculate the cross section on, in eV

        debug : bool
            Debug mode prints out most of the matrices for the first incident energy 

        gamma_matrix : numpy array
            The gamma matrix

        P_matrix : numpy array
            The P matrix

        L_matrix : numpy array
            The L matrix

        L_inv : numpy array
            The inverse of the L matrix

        A_inv : numpy array
            The inverse of the A matrix

        A_matrix : numpy array
            The A matrix

        U_matrix : numpy array
            The U matrix

        total_cross_section : numpy array
            The total cross section for the spin group
            
        Methods
        -------
        set_up_gamma_matrix 
            Function to set up the gamma matrix

        set_up_L_matrix
            Function to set up the L and P matrices

        set_up_A_matrix
            Function to set up the A matrix

        calc_cross_section
            Function to set up the U matrix and calculate the cross
            sections for the channels and for the spin group

        update_gamma_matrix(self,new_gamma_matrix):
            Function to take a new gamma matrix in and update all of 
            the matrices after it, and recalculate the cross section
        
        
        """

        self.res_energies = np.array(res_energies)

        # if only one outgoing channel is provided, convert to a list
        if type(outgoing_channels) != list:
            outgoing_channels = [outgoing_channels]

        # set up all channels, which is icident and all outgoing
        self.channels = [incident_channel] + outgoing_channels

        self.energy_grid = np.array(energy_grid)

        # integer dimension values
        self._Nl = len(self.res_energies)
        self._Nc = len(self.channels)
        self._Ne = len(self.energy_grid)

        # call the methods to calculate the cross sections
        self.set_up_gamma_matrix(debug)
        self.set_up_L_matrix(debug)
        self.set_up_A_matrix(debug)
        self.calc_cross_section(debug)

    def set_up_gamma_matrix(self, debug=False):
        """ Function to set up the gamma matrix
        
        Parameters
        ----------
        debug : bool, optional, default is False

        Returns
        -------
        None
        
        """
        self.gamma_matrix = np.zeros((self._Nl, self._Nc))
        for i, channel in enumerate(self.channels):
            self.gamma_matrix[:,i] = channel.reduced_width_aplitudes
        if debug: print("\n\ngamma: ", self.gamma_matrix.shape, "\n", self.gamma_matrix)

    def set_up_L_matrix(self, debug=False):
        """ Function to set up the L and P matrices
        
        Parameters
        ----------
        debug : bool, optional, default is False

        Returns
        -------
        None
        
        """
        self.P_matrix = np.zeros((self._Ne,  self._Nc, self._Nc))
        for i, channel in enumerate(self.channels):
            self.P_matrix[:,i,i] = channel.calc_penetrability(self.energy_grid)
        self.P_half = np.sqrt(self.P_matrix)
        if debug: print("\n\nP: ", self.P_matrix.shape, "\n", self.P_matrix[0])

        self.L_matrix = 1j * self.P_matrix
        self.L_inv = np.linalg.inv(self.L_matrix)

    def set_up_A_matrix(self, debug=False):
        """ Function to set up the A matrix
        
        Parameters
        ----------
        debug : bool, optional, default is False

        Returns
        -------
        None
        
        """
        res_energy_matrix = np.array([np.diag(self.res_energies)]*self._Ne)
        id = np.ones((self._Ne, self._Nl, self._Nl)) * np.identity(self._Nl)
        neutron_energy_matrix = self.energy_grid.reshape(self._Ne,1,1) * id

        energy_matrix = res_energy_matrix - neutron_energy_matrix
        energy_matrix = energy_matrix.astype(complex)

        self.A_inv = energy_matrix - self.gamma_matrix@self.L_matrix@self.gamma_matrix.T
        self.A_matrix = np.linalg.inv(self.A_inv)
        if debug: print("\n\nA: ", self.A_matrix.shape, "\n", self.A_matrix[0])

    def calc_cross_section(self, debug=False):
        """ Function to set up the U matrix and calculate the 
        cross sections for the channels and for the spin group
        
        Parameters
        ----------
        debug : bool, optional, default is False

        Returns
        -------
        None
        
        """
        id = np.ones((self._Ne, self._Nc, self._Nc)) * np.identity(self._Nc)

        # W
        self.W_matrix = id + 2j * self.P_half@self.gamma_matrix.T@self.A_matrix@self.gamma_matrix@self.P_half
        if debug: print("\n\nW: ", self.W_matrix.shape, "\n", self.W_matrix[0])

        # U
        self.omega_matrix = np.zeros((self._Ne,self._Nc,self._Nc)).astype(complex)
        for i, channel in enumerate(self.channels):
            self.omega_matrix[:,i,i] = np.exp(-1j*channel.calc_rho(self.energy_grid))
        self.U_matrix = self.omega_matrix@self.W_matrix@self.omega_matrix

        # k-sq for in the incident channel
        self.k_sq = (self.channels[0].calc_k(self.energy_grid))**2

        # get total cross section for the spin group
        self.total_cross_section = 10**24 * 2 * np.pi / self.k_sq * (1 - self.U_matrix[:,0,0].real)

        # calculate the cross section for each channel
        for i, channel in enumerate(self.channels):
            channel.calc_cross_section(self.U_matrix,self.k_sq, 0, i )
        

    def update_gamma_matrix(self,new_gamma_matrix):
        """ Function to take a new gamma matrix in and update all of the
            matrices after it, and recalculate the cross section
            
            Parameters
            ----------
            new_gamma_matrix : np.array
                The gamma matrix to replace with. It must be the same shape
                as the original gamma matrix

            Returns
            -------
            None
            
            """
        
        # check that the gamma matrix is the right shape
        if not np.array_equal(self.gamma_matrix.shape, new_gamma_matrix.shape):
            print(f"Shape of new matrix: {new_gamma_matrix.shape}")
            print("The new gamma matrix needs to be the same shape as the old")
            print(f"gamma matrix: {self.gamma_matrix.shape}. No recalculation will be done.")

            return None
        
        # set the new gamma matrix and then call the rest of the functions
        self.gamma_matrix = new_gamma_matrix
        self.set_up_L_matrix()
        self.set_up_A_matrix()
        self.calc_cross_section()