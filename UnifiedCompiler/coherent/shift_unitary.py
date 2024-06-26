import numpy as np

# Magic basis representation
MBU = np.array([
    [1, 0, 0, 1],
    [1j, 0, 0, -1j],
    [0, 1j, 1j, 0],
    [0, 1, -1, 0]
])/np.sqrt(2)

def magic_basis_choi_to_unitary(magic_choivec):
    # Todo: asssert it is unitary
    n_qubit = int(np.log2(magic_choivec.shape[0])/2)
    if n_qubit > 1:
        raise NotImplementedError
    choivec = MBU.conj().T @ magic_choivec
    return np.sqrt(2) * choivec.reshape(2**n_qubit, 2**n_qubit).T

def _single_qubit_shiftunitary(c, eps):
    shiftnormalizedr = [
        [1, -0.08781393818776861, -0.6121513832743527,  -0.7858494742730642], 
            [1, -0.5290573466397314, 0.5587217940691405, -0.638692634057015], 
        [1, 0.638692850018231, 0.5587209471996052, 0.5290579802781218], 
        [1, 0.6744101492084661, 0.3005697385079519, -0.6744099516895313], 
        [1, 0.7858493524069259, -0.6121514123846854, 0.08781482583852211], 
        [1, -0.5039012082066048, 0.7015462135770417, 0.50390126273273], 
        [1, -0.38384509037812775, -0.8431573122710606, 0.3764952766721625]
        ]    
    
    return [magic_basis_choi_to_unitary(np.array([np.sqrt(1-(c*eps)**2), c*eps , c*eps, c*eps])* _r)  for _r in shiftnormalizedr]    


##########################################
# Compile circuit
##########################################

# Magic basis representation
MBU = np.array([
    [1, 0, 0, 1],
    [1j, 0, 0, -1j],
    [0, 1j, 1j, 0],
    [0, 1, -1, 0]
])/np.sqrt(2)

shiftnormalizedr = [
    [1, -0.08781393818776861, -0.6121513832743527,  -0.7858494742730642], 
        [1, -0.5290573466397314, 0.5587217940691405, -0.638692634057015], 
    [1, 0.638692850018231, 0.5587209471996052, 0.5290579802781218], 
    [1, 0.6744101492084661, 0.3005697385079519, -0.6744099516895313], 
    [1, 0.7858493524069259, -0.6121514123846854, 0.08781482583852211], 
    [1, -0.5039012082066048, 0.7015462135770417, 0.50390126273273], 
    [1, -0.38384509037812775, -0.8431573122710606, 0.3764952766721625]
    ]


shiftnormalizedr_depol = [
    [1, -0.5119028548309803, 0.6044882826917328, -0.6103682358251385], 
 [1, 0.5950091083183248, 0.7616927685801398, 0.2564922753436318], 
 [1, -0.34855203268464613, 0.8524547877355049, 0.38965666345927746], 
 [1, 0.9619877167597977, -0.16567705939248534, -0.21709616485403524], 
 [1, 0.29390324612246527, -0.29316883471039196, 0.9097653083478334], 
 [1, 0.06803947143234874, -0.9829656479690841, 0.17073126614631984], 
 [1, -0.2119306636265364, -0.47126404793024623, -0.8561516167964729], 
 [1, 0.4386062632513781,0.4091808605125539, -0.800122221430494], 
    [1, -0.9459071539601508, -0.20358888219371218, 0.2526088342361018]
]

##########################################
# Multi-qubit shiftunitary
##########################################

import itertools
import numpy as np
def multi_qubit_shiftunitary(n_qubit, c, eps):
    """
    n_qubit: number of qubits
    eps: target half-diamond norm
    """
    if n_qubit == 1:
        return _single_qubit_shiftunitary(c, eps)
    theta = np.arcsin(c*eps)
    shift_unitary_array = []
    for (i, j) in itertools.combinations(range(2**n_qubit), 2):
        diag_parts =np.ones(2**n_qubit, dtype = complex)
        diag_parts[i] = np.exp(1j * theta)
        diag_parts[j] = np.exp(-1j * theta)
        shift_unitary_array.append(np.diag(diag_parts))
        
        diag_parts =np.ones(2**n_qubit, dtype = complex)
        diag_parts[i] = np.exp(-1j * theta)
        diag_parts[j] = np.exp(1j * theta)
        shift_unitary_array.append(np.diag(diag_parts))
    return shift_unitary_array    

from scipy.stats import unitary_group
import random

def multi_qubit_randomshiftunitary(n_qubit, c, eps, seed = None, nterm = None):
    """
    n_qubit: number of qubits
    eps: target half-diamond norm
    """
    theta = np.arcsin(c*eps)
    shift_unitary_array = []
    if nterm is None:
        nterm = n_qubit **2
        
    if seed is not None:
        np.random.seed(seed)
    seed_array = np.random.randint(0, 1234567890, size = nterm)
    
    for n in range(nterm):
        diag_parts =np.ones(2**n_qubit, dtype = complex)
        u_haar = unitary_group(dim = 2**n_qubit, seed=seed_array[n]).rvs()
        
        i, j = random.sample(range(2**n_qubit), 2)
        diag_parts[i] = np.exp(1j * theta)
        diag_parts[j] = np.exp(-1j * theta)
        shift_unitary_array.append(u_haar @ np.diag(diag_parts) @ u_haar.conj().T)
        
    return shift_unitary_array   