import src.utils_measure as utils

import numpy as np

measurement = utils.Measurement(utils.Kwiat, 2)

rho_in = np.ones((2,2,2,2))/4.
for i in [0,1,2,3]:
    for j in[0,1,2,3]:
        m, p = measurement.measure(rho_in, [i,j], return_state=True)
        print(i,j,round(m,2))
m_all = np.array([[measurement.measure(rho_in, [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
print(m_all)

print("the same for pure state:")        
psi_in = np.ones((2,2))/2.
for i in [0,1,2,3]:
    for j in[0,1,2,3]:
        m, p = measurement.measure_pure(psi_in, [i,j], return_state=True)
        print(i,j,round(m,2))
        
print("now do the tomography:")
tomography = utils.Tomography(2, utils.Kwiat_projectors)
tomography.calulate_B_inv()
rho_rec = tomography.reconstruct(m_all)
print(np.round(rho_rec, 2))

# test for Bell state:
rho_in = np.zeros((4,4))
rho_in[0,0] = 0.5
rho_in[0,3] = 0.5
rho_in[3,0] = 0.5
rho_in[3,3] = 0.5
m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
rho_rec = tomography.reconstruct(m_all)
print(np.round(rho_rec, 2))

# test for custom state:
print("tomograpy for custom state")
rho_in = np.load('./test_states/dens0.npy')
print(np.round(rho_in, 2))
m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
rho_rec = tomography.reconstruct(m_all, enforce_positiv_sem=True)
print(np.round(rho_rec, 2))
print(rho_in-rho_rec)
print(np.trace(rho_rec))
print(np.all(np.linalg.eigvals(rho_rec) > 0))
