import numpy as np
from UnifiedCompiler.utils.channel_utils import unitary_to_ptm
from UnifiedCompiler.utils.channel_utils import idx_to_pauli_mat, operate_channel
from UnifiedCompiler.utils.channel_utils import choi_to_ptm


def transition_matrix(vector1, vector2):
    return np.outer(vector1, vector2.conj())

def ptm_list_to_Amatrix(ptm_list):
    Amat = []
    n_qubit = int(np.log2(ptm_list[0].shape[0]))//2
    for _ptm in ptm_list:
        Amat.append(_ptm.reshape(_ptm.size))
    return np.array(Amat).T

def maximally_entangled_state(n_qubit):
    return np.identity(2**n_qubit).reshape(2**(2*n_qubit))

def _tuple_operation_to_choi(tup):
    n_qubit = int(np.log2(tup[0].shape[0]))
    mes = maximally_entangled_state(n_qubit)
    rho = np.outer(mes, mes.conj())

    tup_extend = (np.kron(np.identity(2**n_qubit), tup[0]), np.kron(np.identity(2**n_qubit), tup[1]))
    rho = tup_extend[0] @ rho @ tup_extend[1]
    return rho

def _tuple_operation_to_ptm(tup):
    return choi_to_ptm(_tuple_operation_to_choi(tup))

def get_pauli_transfer_matrix(channel, n_qubit : int):
        if type(channel) == tuple:
            n_qubit = int(np.log2(channel[0].shape[0]))
            return _tuple_operation_to_ptm(channel)
            #Amat.append(.reshape(16**n_qubit))
        elif type(channel) == np.ndarray:
            from qulacs import QuantumCircuit as QC
            from UnifiedCompiler.utils.channel_utils import circuit_to_ptm
            n_qubit = int(np.log2(channel.shape[0]))
            _qc = QC(n_qubit)
            _qc.add_dense_matrix_gate(list(range(n_qubit)), channel)
            return circuit_to_ptm(_qc)
        else:
            return _get_pauli_transfer_matrix_slow(channel, n_qubit)
            #Amat.append(.reshape(16**n_qubit))
    

def _get_pauli_transfer_matrix_slow(channel, n_qubit : int):
    ptm = np.zeros((4**n_qubit, 4**n_qubit), dtype = complex)
    for jj in range(4**n_qubit):
        Pjj = idx_to_pauli_mat(jj, n_qubit)
        C_Pjj = operate_channel(channel, Pjj)
        
        for j in range(4**n_qubit):
            Pj = idx_to_pauli_mat(j, n_qubit)
            ptm[j, jj] = np.trace(Pj @ C_Pjj)/2**n_qubit
            
    return ptm

iden = np.array([[1, 0], [0, 1]])
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
hadamard = np.array([[1, 1], [1, -1]])/np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]])
S = np.array([[1, 0], [0, np.exp(1j * np.pi/2)]])
Sdag = np.array([[1, 0], [0, np.exp(-1j * np.pi/2)]])

single_paulis = [iden, sigmax, sigmay, sigmaz,]
ptm_single_paulis = [unitary_to_ptm(_p) for _p in single_paulis]

hadamard_set = [(iden + 1j * sigmax)/np.sqrt(2), (iden + 1j * sigmay)/np.sqrt(2), (iden + 1j * sigmaz)/np.sqrt(2)]
hadamard_set += [(sigmay + sigmaz)/np.sqrt(2), (sigmaz + sigmax)/np.sqrt(2), (sigmax + sigmay)/np.sqrt(2), ]
ptm_hadamards = [unitary_to_ptm(_u) for _u in hadamard_set]

# projection operations
proj_set = [(iden + sigmax)/2, (iden + sigmay)/2, (iden + sigmaz)/2,]
proj_set += [(sigmay + 1j * sigmaz)/2, (sigmaz + 1j * sigmax)/2, (sigmax + 1j * sigmay)/2,]
stochastic_set = [(iden + sigmax + sigmay)/3, (sigmax + sigmay + sigmaz)/3]

# preparation operation
zero_state = np.array([1, 0])
one_state = np.array([0, 1])
Xplus_state = np.array([1, 1])/np.sqrt(2)
Yplus_state = np.array([1, +1j])/np.sqrt(2)
Zplus_state = np.array([1, 0])

prep_set = [
    [(transition_matrix(Xplus_state, zero_state), transition_matrix(zero_state, Xplus_state)), 
     (transition_matrix(Xplus_state, one_state), transition_matrix(one_state, Xplus_state, ))], #prepare |+X>
    [(transition_matrix(Yplus_state, zero_state), transition_matrix(zero_state, Yplus_state)), 
     (transition_matrix(Yplus_state, one_state), transition_matrix(one_state, Yplus_state, ))], #prepare |+Y>
    [(transition_matrix(Zplus_state, zero_state), transition_matrix(zero_state, Zplus_state, )), 
     (transition_matrix(Zplus_state, one_state), transition_matrix(one_state, Zplus_state, ))], #prepare |+Z>    
]
ptm_preps = [get_pauli_transfer_matrix(ch, 1) for ch in prep_set]

#universal_set = single_paulis + hadamard_set + proj_set # Endo PRX 2018
universal_set = single_paulis + hadamard_set + prep_set # Takagi PRR 2021
universal_ptm_list = ptm_single_paulis + ptm_hadamards + ptm_preps

# Ordinary PEC
A_mat_pauli = ptm_list_to_Amatrix(ptm_single_paulis)
A_mat_univ = ptm_list_to_Amatrix(universal_ptm_list)