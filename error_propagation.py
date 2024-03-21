import os
import numpy as np
import matplotlib.pyplot as plt

import utils_measure as utils

val_set = "../matrices"

measurement = utils.Measurement(utils.Kwiat, 2)
tomography = utils.Tomography(2, utils.Kwiat_projectors)
tomography.calulate_B_inv()
"""
# noise in rho -> noise in m=?
for w in [0,1,2,3]:
    for k in [0,1,2,3]:
        m_err = []
        for i, file in enumerate(os.listdir(val_set)):
            if file.endswith(".npy"):
                rho_in = np.load(os.path.join(val_set, file))
                m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
                noiss = np.random.normal(0., .01)
                rho_in[w,k] += noiss
                rho_in[k,w] += noiss
                m_all_err = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
                m_err.append((m_all-m_all_err).reshape((4,4)))
            if i > 1000: break
        m_err = np.array(m_err).transpose((1,2,0))
        m_std = np.std(m_err, axis=2)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ims = ax.imshow(m_std, interpolation='none')
        fig.colorbar(ims)
        fig.savefig('./results/err_m_'+str(w)+'_'+str(k)+'.png') 
        plt.close(fig)

# noise in m -> noise in rho=?
for w in [0,1,2,3]:
    for k in [0,1,2,3]:
        rho_err = []
        for i, file in enumerate(os.listdir(val_set)):
            if file.endswith(".npy"):
                rho_in = np.load(os.path.join(val_set, file))
                m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
                m_all[w*4+k] += np.random.normal(0., .01)
                rho_rec = tomography.reconstruct(m_all)
                rho_err.append((rho_in-rho_rec))
            if i > 1000: break
        rho_err = np.array(rho_err).transpose((1,2,0))
        rho_std = np.std(rho_err, axis=2)
        # plot
        fig, ax = plt.subplots()
        ims = ax.imshow(rho_std, interpolation='none')
        fig.colorbar(ims)
        fig.savefig('./results/err_r_'+str(w)+'_'+str(k)+'.png') 
        plt.close(fig)
"""

Xs = utils.X_states(std=2.)

# noise in X -> noise in m=?
m_err = []
for i in range(1000):
    rho_in = Xs.generate()
    #rho_in = Xs.generate(A3=0., B3=0., C1=1.0, C2=-1.0, C3=1.0)  # Bell
    m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
    rho_in = Xs.add_noise_r(rho_in, std=.01)
    m_all_err = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
    m_err.append((m_all-m_all_err).reshape((4,4)))
    print(Xs.concurrence(rho_in))
m_err = np.array(m_err).transpose((1,2,0))
m_std = np.std(m_err, axis=2)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ims = ax.imshow(m_std, interpolation='none')
fig.colorbar(ims)
fig.savefig('./results/Xerr_m.png') 
plt.close(fig)

# noise in m -> noise in X=?        
for w in [0,1,2,3]:
    for k in [0,1,2,3]:
        rho_err = []
        for _ in range(1000):
            rho_in = Xs.generate()
            m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
            m_all[w*4+k] += np.random.normal(0., .05)
            rho_rec = tomography.reconstruct(m_all)
            rho_err.append((rho_in-rho_rec))
        rho_err = np.array(rho_err).transpose((1,2,0))
        rho_std = np.std(rho_err, axis=2)
        # plot
        fig, ax = plt.subplots()
        ims = ax.imshow(rho_std, interpolation='none')
        fig.colorbar(ims)
        fig.savefig('./results/Xerr_r_'+str(w)+'_'+str(k)+'.png') 
        plt.close(fig)
