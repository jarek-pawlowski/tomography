import os
import numpy as np

import utils_measure as utils


ms = utils.Measurement(utils.Kwiat, 2)

directory = './derandomized_test/single'
os.makedirs(directory, exist_ok=True)
test_states = [utils.tensordot(ms.basis[i], ms.basis[j], indices=0).reshape((4,4)) for j in [0,1,2,3] for i in [0,1,2,3]]
for i, state in enumerate(test_states):
    np.save(os.path.join(directory, 'm_'+str(i)), state)

directory = './derandomized_test/double'
os.makedirs(directory, exist_ok=True)
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        for i1 in [0,1,2,3]:
            for j1 in [0,1,2,3]:
                state = utils.tensordot(ms.basis[i], ms.basis[j], indices=0) + utils.tensordot(ms.basis[i1], ms.basis[j1], indices=0)
                np.save(os.path.join(directory, 'm_'+str(i*4+j)+'_'+str(i1*4+j1)), state.reshape((4,4))/2.)

directory = './derandomized_test/Xs'
os.makedirs(directory, exist_ok=True)
Xs = utils.X_states(std=2.)
for i in range(1000):
    np.save(os.path.join(directory, str(i)), Xs.generate().reshape((4,4)))

directory = './derandomized_test/Bell'
os.makedirs(directory, exist_ok=True)
np.save(os.path.join(directory, '0'), Xs.generate(A3=0., B3=0., C1=1.0, C2=-1.0, C3=1.0).reshape((4,4)))

