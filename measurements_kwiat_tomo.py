import utils_measure as utils

import numpy as np

measurement = utils.Measurement(utils.Kwiat, 2)
Kwiat_code = utils.Kwiat_library(utils.basis_for_Kwiat_code)
breakpoint()

rho_in = np.ones((2,2,2,2))/4.
m_all = np.array([[measurement.measure(rho_in, [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
rho_rec = Kwiat_code.run_tomography(m_all, method='MLE')  # possible methods are: 'MLE', 'HMLE', 'LINEAR'
print(np.round(rho_rec, 2)) 

# test for Bell state:
rho_in = np.zeros((4,4))
rho_in[0,0] = 0.5
rho_in[0,3] = 0.5
rho_in[3,0] = 0.5
rho_in[3,3] = 0.5
m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
rho_rec = Kwiat_code.run_tomography(m_all, method='MLE')
print(np.round(rho_rec, 2))

# now test for Bell state with non-ideal measurements, MLE method
# https://research.physics.illinois.edu/QI/Photonics/tomography-files/amo_tomo_chapter.pdf
intensity = np.ones(len(utils.basis_for_Kwiat_code))
intensity[5:] = 0.8
rho_rec = Kwiat_code.run_tomography(m_all, intensity=intensity, method='MLE')  # possible methods are: 'MLE', 'HMLE', 'LINEAR'
print(np.round(rho_rec, 2))

# other technique (probably better for limited measuerments): 
# MaxEnt: http://www.quantum.physics.sk/rcqi/research/publications/2004/2004-5.pdf

# test for custom state:
print("tomograpy for custom state")
rho_in = np.load('./test_states/dens0.npy')
print(np.round(rho_in, 2))
m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
rho_rec = Kwiat_code.run_tomography(m_all, method='MLE')
print(np.round(rho_rec, 2))
print(rho_in-rho_rec)
