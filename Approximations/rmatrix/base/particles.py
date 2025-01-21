class Particle:
    def __init__(self,label,A,Z,Sn=0):
        """ Class to hold information about a particle/isotope
        
        Parameters
        ----------
        label : str
            string label for the particle

        A : int
            Mass number

        Z : int
            Z of the particle

        Sn : float, optional, default: 0
            Neutron separation energy in eV, if applicable
        
        Attributes
        ----------
        label : str
            string label for the particle

        A : int
            Mass number

        Z : int
            Z of the particle

        Sn : float, optional, default: 0
            Neutron separation energy in eV, if applicable
        
        Methods
        -------
        No methods are currently defined
        """

        self.label = label
        self.A = A
        self.Z = Z
        self.Sn = Sn

    def __repr__(self):
        return f"{self.label}"