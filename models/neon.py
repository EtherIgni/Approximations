from models import basic
from rmatrix import Particle, ElasticChannel, CaptureChannel, SpinGroup
import numpy as np

model=basic.base_reaction()

neutron = Particle('n',1,0)
gamma = Particle('g',0,0)
target = Particle("20Ne",20,10)
compound = Particle("20Ne", 20,10, Sn=6.6e6)
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
excitation = 5e5  # the product is left in the 1st ex state at 0.5MeV
model.add_capture_channel(J,pi,ell,radius,reduced_width_amplitudes,excitation)

res_energies = [1e6,1.1e6]
energy_grid = np.linspace(0.9e6,1.2e6,1001)
model.set_resonance_energies(res_energies)
model.set_energy_grid(energy_grid)

model.establish_spin_group()