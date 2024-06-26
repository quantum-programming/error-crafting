#import qulacs
#from qulacs import QuantumCircuit, QuantumState
from qulacs import QuantumCircuit as QC
from qulacs import QuantumState
import numpy as np


def unitary_to_choi(unitary):
    n_qubit = int(np.log2(unitary.shape[0]))    
    max_entangled_vec = np.identity(2**n_qubit).reshape(2**(2*n_qubit))
    state = QuantumState(2*n_qubit)
    state.load(max_entangled_vec)    
    
    
    large_circuit = QC(2*n_qubit)
    large_circuit.add_dense_matrix_gate(list(range(n_qubit)), unitary)            
    large_circuit.update_quantum_state(state)
    vec = state.get_vector()
    
    return np.outer(vec, vec.conj())    

def to_channel(unitary):
    import qutip as qt
    return qt.to_super(qt.Qobj(unitary))

#import qiskit


##################################################
# diamond norms
################################################

from scipy import sparse
from qiskit.quantum_info import Choi
def diamond_norm(data1, data2, is_unitary = False, **kwargs):
    import cvxpy
    
    if is_unitary:
        ptm1 = unitary_to_ptm(data1)
        ptm2 = unitary_to_ptm(data2)
    else:
        ptm1 = data1.copy()
        ptm2 = data2.copy()
    assert np.isclose(np.log2(ptm1.shape[0])%2, 0), "Maybe data1 is unitary? then set is_unitary =True."

    
    choi = Choi(ptm_to_choi(ptm1 - ptm2))
    
    from scipy import sparse

    #cvxpy = _cvxpy_check("`diamond_norm`")  # Check CVXPY is installed

    #choi = Choi(_input_formatter(choi, Choi, "diamond_norm", "choi"))

    def cvx_bmat(mat_r, mat_i):
        """Block matrix for embedding complex matrix in reals"""
        return cvxpy.bmat([[mat_r, -mat_i], [mat_i, mat_r]])

    # Dimension of input and output spaces
    dim_in = choi._input_dim
    dim_out = choi._output_dim
    size = dim_in * dim_out

    # SDP Variables to convert to real valued problem
    r0_r = cvxpy.Variable((dim_in, dim_in))
    r0_i = cvxpy.Variable((dim_in, dim_in))
    r0 = cvx_bmat(r0_r, r0_i)

    r1_r = cvxpy.Variable((dim_in, dim_in))
    r1_i = cvxpy.Variable((dim_in, dim_in))
    r1 = cvx_bmat(r1_r, r1_i)

    x_r = cvxpy.Variable((size, size))
    x_i = cvxpy.Variable((size, size))
    iden = sparse.eye(dim_out)


    # Watrous uses row-vec convention for his Choi matrix while we use
    # col-vec. It turns out row-vec convention is requried for CVXPY too
    # since the cvxpy.kron function must have a constant as its first argument.
    c_r = cvxpy.bmat([[cvxpy.kron(iden, r0_r), x_r], [x_r.T, cvxpy.kron(iden, r1_r)]])
    c_i = cvxpy.bmat([[cvxpy.kron(iden, r0_i), x_i], [-x_i.T, cvxpy.kron(iden, r1_i)]])
    c = cvx_bmat(c_r, c_i)

    # Convert col-vec convention Choi-matrix to row-vec convention and
    # then take Transpose: Choi_C -> Choi_R.T
    choi_rt = np.transpose(
        np.reshape(choi.data, (dim_in, dim_out, dim_in, dim_out)), (3, 2, 1, 0)
    ).reshape(choi.data.shape)
    choi_rt_r = choi_rt.real
    choi_rt_i = choi_rt.imag

    # Constraints
    cons = [
        r0 >> 0,
        r0_r == r0_r.T,
        r0_i == -r0_i.T,
        cvxpy.trace(r0_r) == 1,
        r1 >> 0,
        r1_r == r1_r.T,
        r1_i == -r1_i.T,
        cvxpy.trace(r1_r) == 1,
        c >> 0,
    ]

    # Objective function
    obj = cvxpy.Maximize(cvxpy.trace(choi_rt_r @ x_r) + cvxpy.trace(choi_rt_i @ x_i))
    prob = cvxpy.Problem(obj, cons)

    # Solve the problem using MOSEK solver
    sol = prob.solve(solver=cvxpy.MOSEK, **kwargs)

    return sol
from qiskit.quantum_info import PTM, Choi, Operator
from qiskit.quantum_info import diamond_norm as dnorm
import copy
def diamond_norm_precise(data1, data2=None, scale = 1e9, is_unitary = False):
    if is_unitary:
        #ptm1 = unitary_to_ptm(data1)
        #ptm2 = unitary_to_ptm(data2)
        choi1 = unitary_to_choi(data1)
        choi2 = unitary_to_choi(data2)
        return np.abs(np.linalg.svd(choi1 - choi2)[1]).sum()/2
    else:
        ptm1 = copy.deepcopy(data1)
        ptm2 = copy.deepcopy(data2)

    if ptm2 is None:
        ptm2 = np.diag(np.ones(4, dtype = complex))
    return dnorm(PTM((ptm1 - ptm2)* scale) )/scale
    #else:
        #return dnorm(PTM((ptm1 - ptm2)*scale) )/scale

def diamond_norm_from_choi(ptm1, ptm2=None):
    if ptm2 is None:
        ptm2 = unitary_to_ptm(np.diag([1,1]))
    #else:
    choi1, choi2 = ptm_to_choi(ptm1), ptm_to_choi(ptm2)
    return np.linalg.svd(choi1 - choi2)[1].sum()/2

#from UnifiedCompiler.utils.channel_utils import ptm_to_choi_MB

#choi_mb = ptm_to_choi_MB(error_ptm_list[0])

def ptm_to_mb_vec(ptm):
    #choi_mb ptm_to_choi_MB(error_ptm_list[0])
    vals, vecs = np.linalg.eigh(ptm_to_choi_MB(ptm))
    mbvec = vecs[:, np.argmax(vals)]
    assert np.allclose(np.sort(vals)[::-1][1:], 0), "input is not unitary."
    return mbvec    

def diamond_norm_older(operation1, operation2, ):

    if "qutip" in str(type(operation1)):
        ch1 = operation1 
        ch2 = operation2
    else:
        ch1 = to_channel(operation1)
        ch2 = to_channel(operation2)
    return (ch1 - ch2).dnorm()
    
def l1_norm(operation1, operation2):
    import qutip as qt
    n_qubit = count_num_qubits(operation1)
    if "qutip" in str(type(operation1)):
        ch1 = operation1
        ch2 = operation2
    else:
        ch1 = to_channel(operation1)
        ch2 = to_channel(operation2)
    choi1 = (qt.to_choi(ch1)).full()
    choi2 = (qt.to_choi(ch2)).full()

    ptm1 = choi_to_ptm(choi1)
    ptm2 = choi_to_ptm(choi2)
    return np.abs(ptm1 - ptm2).sum()

def _operator_norm(matrix):
    # Compute the singular values of the matrix
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    
    # The operator norm is the maximum singular value
    return np.max(singular_values)

def operator_norm(A, B = None):
    if B is not None:
        return _operator_norm(A - B)
    return _operator_norm(A)

def count_num_qubits(obj):
    if type(obj) == np.ndarray:
        return int(np.log2(obj.shape[0]))

    elif "qutip" in str(type(obj)):
        if obj.issuper:
            return len(obj.dims[0][0])
        elif obj.isoper:
            return len(obj.dims[0])

    raise TypeError

######################################################
# Channel calculation
######################################################

from qulacs import DensityMatrix, ParametricQuantumCircuit

def circuit_to_choi(circuit: QC):
    n_qubit = circuit.get_qubit_count()
    max_entangled_vec = np.identity(2**n_qubit).reshape(2**(2*n_qubit))
    state = DensityMatrix(2*n_qubit)
    state.load(max_entangled_vec)

    large_map = ParametricQuantumCircuit(2*n_qubit)
    for i in range(circuit.get_gate_count()):
        large_map.add_gate(circuit.get_gate(i))

    large_map.update_quantum_state(state)
    return state.get_matrix()

def circuit_to_choi_matrix(circuit: QC):
    return circuit_to_choi(circuit)

def circuit_to_ptm(circuit):
    choi_mat = circuit_to_choi(circuit)
    return choi_matrix_to_ptm(choi_mat)

def circuit_to_ptm_unitary(circuit):
    # Todo: assert unitarity
    choi = circuit_to_choi_unitary(circuit)
    return choi_to_ptm(choi)

def circuit_to_choi_unitary(circuit):
    # ToDo: assert unitarity
    n_qubit = circuit.get_qubit_count()
    max_entangled_vec = np.identity(2**n_qubit).reshape(2**(2*n_qubit))
    state = QuantumState(2*n_qubit)
    state.load(max_entangled_vec)    
    
    large_circuit = QC(2*n_qubit)
    
    for i in range(circuit.get_gate_count()):
        large_circuit.add_gate(circuit.get_gate(i))
        
    large_circuit.update_quantum_state(state)
    vec = state.get_vector()
    
    return np.outer(vec, vec.conj())

def unitary_to_choi(unitary):
    n_qubit = int(np.log2(unitary.shape[0]))    
    max_entangled_vec = np.identity(2**n_qubit).reshape(2**(2*n_qubit))
    state = QuantumState(2*n_qubit)
    state.load(max_entangled_vec)    
    
    
    large_circuit = QC(2*n_qubit)
    large_circuit.add_dense_matrix_gate(list(range(n_qubit)), unitary)            
    large_circuit.update_quantum_state(state)
    vec = state.get_vector()
    
    return np.outer(vec, vec.conj())    

def unitary_to_ptm(unitary):
    choi = unitary_to_choi(unitary)
    return choi_to_ptm(choi)
    
def _preprocess_choi_matrix(choi_matrix):
    n_qubit = int(np.log2(choi_matrix.shape[0])/2)
    choi_tensor = choi_matrix.reshape(2**n_qubit, 2**n_qubit, 2**n_qubit, 2**n_qubit)
    choi_tensor_new = np.einsum("ijkl->ljik", choi_tensor)
    return choi_tensor_new.reshape(4**n_qubit, 4**n_qubit)

##################################
# Canonical and Magic-basis Choi matrix
##################################

# Magic basis representation
MBU = np.array([
    [1, 0, 0, 1],
    [1j, 0, 0, -1j],
    [0, 1j, 1j, 0],
    [0, 1, -1, 0]
])/np.sqrt(2)

def canonical_to_magic_basis_choi_matrix(choi_matrix):
    n_qubit = int(np.log2(choi_matrix.shape[0])//2)
    if n_qubit > 1:
        raise NotImplementedError
    return MBU @ choi_matrix @ MBU.conj().T

def magic_to_canonical_basis_choi_matrix(choi_MB):
    n_qubit = int(np.log2(choi_MB.shape[0])//2)
    if n_qubit > 1:
        raise NotImplementedError
    return MBU.conj().T @ choi_MB @ MBU

def magic_basis_choi_to_unitary(magic_choivec):
    # Todo: asssert it is unitary
    n_qubit = int(np.log2(magic_choivec.shape[0])/2)
    if n_qubit > 1:
        raise NotImplementedError
    choivec = MBU.conj().T @ magic_choivec
    return np.sqrt(2) * choivec.reshape(2**n_qubit, 2**n_qubit).T

def choi_MB_to_ptm(choi_MB):
    choi_CB = magic_to_canonical_basis_choi_matrix(choi_MB)
    return choi_to_ptm(choi_CB)

def ptm_to_choi_MB(ptm:np.ndarray):
    choi_CB = ptm_to_choi(ptm)
    return canonical_to_magic_basis_choi_matrix(choi_CB)

#################################
# PTM - Choi conversion
#################################
import numpy as np
from openfermion import QubitOperator, get_sparse_operator

def circuit_to_ptm_bydefinition(channel, n_qubit : int):
    ptm = np.zeros((4**n_qubit, 4**n_qubit), dtype = complex)
    for jj in range(4**n_qubit):
        Pjj = idx_to_pauli_mat(jj, n_qubit)
        C_Pjj = operate_channel(channel, Pjj)
        
        for j in range(4**n_qubit):
            Pj = idx_to_pauli_mat(j, n_qubit)
            ptm[j, jj] = np.trace(Pj @ C_Pjj)/2**n_qubit
            
    return ptm

def get_pauli_coeff(operator):
    n_qubit = int(np.log2(operator.shape[0]))
    q = np.zeros(4**n_qubit, dtype = complex)
    for ii in range(4**n_qubit):
        Pii = idx_to_pauli_mat(ii, n_qubit)
        q[ii] =np.trace(Pii @ operator)/2**n_qubit
    return q

def choi_matrix_to_ptm(choi_matrix):
    n_qubit = int(np.log2(choi_matrix.shape[0])/2)
    #beta_tensor = create_beta_tensor(n_qubit)
    beta_matrix = create_beta_matrix(n_qubit)

    # rewrite {ii' jj'} → {ij j'i'}
    choi_tensor = choi_matrix.reshape(2**n_qubit, 2**n_qubit, 2**n_qubit, 2**n_qubit)
    choi_tensor_new = np.einsum("ijkl->ljik", choi_tensor)
    choi_matrix_new = choi_tensor_new.reshape(4**n_qubit, 4**n_qubit)
    return (beta_matrix @ choi_matrix_new @ (beta_matrix.T))/2**n_qubit

def choi_to_ptm(choi_matrix):
    assert len(choi_matrix.shape) == 2
    return choi_matrix_to_ptm(choi_matrix)

def ptm_to_choi_matrix(ptm):
    n_qubit = int(np.log2(ptm.shape[0])/2)
    #beta_tensor = create_beta_tensor(n_qubit)
    beta_matrix = create_beta_matrix(n_qubit)
    beta_inv = np.linalg.inv(beta_matrix)

    choi_matrix_conv = beta_inv @ (ptm.T * (2**n_qubit)) @ beta_inv.T
    
    # rewrite {ij j'i'} → {ii' jj'}
    choi_tensor_conv = choi_matrix_conv.reshape(2**n_qubit, 2**n_qubit, 2**n_qubit, 2**n_qubit)
    choi_tensor = np.einsum("iklj->ijkl", choi_tensor_conv)
    choi_matrix = choi_tensor.reshape(4**n_qubit, 4**n_qubit)
    return choi_matrix

def ptm_to_choi(ptm):
    return ptm_to_choi_matrix(ptm)


iden = np.array([[1, 0], [0, 1]])
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
single_paulis = [iden, sigmax, sigmay, sigmaz,]

beta_tensor_1q = np.zeros((4, 2, 2), dtype = complex)
for i in range(4):
    _pauli_mat = single_paulis[i]
    beta_tensor_1q[i, :, :] = _pauli_mat
    
beta_matrix_1q = np.zeros((4, 4), dtype = complex)
for i in range(4):
    _pauli_mat = single_paulis[i]
    beta_matrix_1q[i, :] = _pauli_mat.reshape(_pauli_mat.size)    
    
def get_diff_list(i, j, n_qubit):
    i_quartstring = np.base_repr(i, 4).zfill(n_qubit)
    j_quartstring = np.base_repr(j, 4).zfill(n_qubit)
    i_pauli_id_list = [int(s) for s in i_quartstring]
    j_pauli_id_list = [int(s) for s in j_quartstring]

    diff_list = []
    for k, (pi, pj) in enumerate(zip(i_pauli_id_list, j_pauli_id_list)):
        if pi != pj:
            diff_list.append((k, pi, pj))
    return diff_list

def update_pauli_sp(i, j, n_qubit, pauli_sparse):
    global local_pauli_sp_data
    diff_list = get_diff_list(i, j, n_qubit)
    for (k, pi, pj) in diff_list:
        if pi != 0:
            label_before = f"{pauli_strings[pi]}{k}"
            pauli_sparse = pauli_sparse * local_pauli_sp_data[label_before]
        if pj != 0:
            label_after = f"{pauli_strings[pj]}{k}"
            pauli_sparse = pauli_sparse * local_pauli_sp_data[label_after]
    return pauli_sparse

pauli_strings = ["I", "X", "Y", "Z"]
local_pauli_sp_data = {}
def initialize_local_pauli_sp_data(n_qubit):
    global local_pauli_sp_data
    local_pauli_sp_data["n_qubit"] = n_qubit
    for i in range(n_qubit):
        for a in [1, 2, 3]:
            label = f"{pauli_strings[a]}{i}"
            local_pauli_sp_data[label] = get_sparse_operator(QubitOperator(label), n_qubits = n_qubit)    

def tensor_product(matrix_list):
    ret = matrix_list[0]
    for mat in matrix_list[1:]:
        ret = np.kron(ret, mat)
    return ret

def create_beta_tensor(n_qubit):
    iden = np.diag([1, 1])
    sigmax = np.array([[0, 1,], [1, 0]])
    sigmay = np.array([[0, -1j,], [1j, 0]])
    sigmaz = np.diag([1,-1])
    single_paulis = [iden, sigmax, sigmay, sigmaz]
    return np.array([tensor_product([single_paulis[int(a)] for a in np.base_repr(i, base = 4).zfill(n_qubit)]) for i in range(4**n_qubit)])    

def create_beta_tensor_old(n_qubit):
    global local_pauli_sp_data

    if n_qubit == 1:
        return np.array([[[ 1.+0.j,  0.+0.j],
        [ 0.+0.j,  1.+0.j]],

       [[ 0.+0.j,  1.+0.j],
        [ 1.+0.j,  0.+0.j]],

       [[ 0.+0.j,  0.-1.j],
        [ 0.+1.j,  0.+0.j]],

       [[ 1.+0.j,  0.+0.j],
        [ 0.+0.j, -1.+0.j]]])
    
    
    if not "n_qubit" in local_pauli_sp_data.keys():
        initialize_local_pauli_sp_data(n_qubit)
    elif local_pauli_sp_data["n_qubit"] != n_qubit:
        initialize_local_pauli_sp_data(n_qubit)
    # Todo : make this faster using tensor product
    #beta_local_array = [beta_tensor_1q for _ in range(n_qubit)]
    #ret = np.copy(beta_local_array[0])
    #for _ in range(1, n_qubit):
        #ret = np.kron(ret, beta_tensor_1q)
    #return ret
    
    beta_tensor = np.zeros((4**n_qubit, 2**n_qubit, 2**n_qubit), dtype = complex)
    pauli_sp = get_sparse_operator(QubitOperator(""), n_qubits = n_qubit)
    for i in range(4**n_qubit):
        beta_tensor[i, :, :] = np.copy(pauli_sp.toarray())
        pauli_sp = update_pauli_sp(i, i+1, n_qubit, pauli_sp, )
    return beta_tensor
    

def create_beta_matrix(n_qubit):
    return create_beta_tensor(n_qubit).reshape(4**n_qubit, 4**n_qubit)


#######################################
# other utils
#######################################

from qulacs import DensityMatrix
from qulacs.observable import create_observable_from_openfermion_text
from openfermion import QubitOperator, get_sparse_operator
import numpy as np

def _pauli(pauli_id, qubit_pos):
    if pauli_id ==0:
        return QubitOperator("")
    else:
        return QubitOperator("%s%d"%(["I", "X", "Y", "Z"][pauli_id], qubit_pos))    

def pauli(pauli_id_list, qubit_pos_list):
    """
    attributes:
        pauli_id: List of pauli ids, or int
        qubit_pos: List of index of qubit, or int
    """
    if type(pauli_id_list) == int and type(qubit_pos_list) == int:
        return _pauli(pauli_id_list, qubit_pos_list)
    else:
        assert type(pauli_id_list) == list and type(qubit_pos_list) == list
        ret = QubitOperator("")
        for _pauli_id, _qubit_pos in zip(pauli_id_list, qubit_pos_list):
            ret = _pauli(_pauli_id, _qubit_pos) * ret
        return ret

def expectation(operator, state):
    if str(type(operator)).split(".")[0][8:] == "openfermion":
        n_qubit = int(np.log2(state.shape[0]))
        operator = get_sparse_operator(operator, n_qubits = n_qubit)
    if len(state.shape) == 1:
        return state.conj() @ operator @ state
    elif len(state.shape) == 2:
        return np.trace(operator @ state)        

def idx_to_pauli(i, n_qubit):
    quartstring = np.base_repr(i, 4).zfill(n_qubit)
    pauli_id_list = [int(s) for s in quartstring]
    #print(i, pauli(pauli_id_list, list(range(n_qubit))))
    return pauli(pauli_id_list, list(range(n_qubit)))

def idx_to_pauli_mat(i, n_qubit):
    quartstring = np.base_repr(i, 4).zfill(n_qubit)
    pauli_id_list = [int(s) for s in quartstring]
    #print(i, pauli(pauli_id_list, list(range(n_qubit))))
    mat = get_sparse_operator(pauli(pauli_id_list, list(range(n_qubit))), n_qubits = n_qubit).toarray()
    return mat

def idx_to_pauli_sp(i, n_qubit):
    quartstring = np.base_repr(i, 4).zfill(n_qubit)
    pauli_id_list = [int(s) for s in quartstring]
    #print(i, pauli(pauli_id_list, list(range(n_qubit))))
    mat = get_sparse_operator(pauli(pauli_id_list, list(range(n_qubit))), n_qubits = n_qubit)
    return mat

def operate_channel(operation, state):
    if type(state) ==np.ndarray:
        n_qubit = int(np.log2(state.shape[0]))
        
    if type(operation) == list or type(operation) == tuple:
        if type(operation[0]) == list or type(operation[0]) == tuple:
            output = sum([_op[0] @ state @ _op[1] for _op in operation])
        else:
            output = operation[0] @ state @ operation[1]
        
    elif type(operation) == np.ndarray:
        output = operation @ state @ (operation.T.conj())
        
    elif "qulacs" in str(type(operation)):
        dm = DensityMatrix(n_qubit)
        dm.load(state)
        operation.update_quantum_state(dm)
        output = dm.get_matrix()
    else:
        raise TypeError()
        
    return output


#######################################
# partial error correction
#######################################

from qulacs import QuantumCircuit
from qulacs.gate import Measurement, P0, P1
from UnifiedCompiler.utils.ptrace_utils import take_ptrace_ptm
def get_X_corrected_ptm(data, qem_type = "FB", is_unitary = False, normalize_ptm = True):
    """
    get the PTM of target unitary/ptm when X-error detection or correction is performed.
    """
    assert qem_type in ["FB", "detection"]
    if is_unitary:
        return _get_X_corrected_ptm_from_unitary(data, qem_type = qem_type, normalize_ptm = normalize_ptm)
    else:
        return _get_X_corrected_ptm(data, qem_type = qem_type, normalize_ptm = normalize_ptm)

def _get_X_corrected_ptm_from_unitary(unitary, qem_type = "FB", normalize_ptm = True):
    assert qem_type in ["FB", "detection"]
    n_qubit = unitary.shape[0].bit_length() - 1
    assert n_qubit == 1
    system, ancilla = [0, 1]
    
    qc = QuantumCircuit(n_qubit + 1) # 1 for ancilla

    qc.add_H_gate(ancilla)
    qc.add_CZ_gate(ancilla, system)

    qc.add_dense_matrix_gate(range(n_qubit), unitary)

    qc.add_CZ_gate(ancilla, system)
    qc.add_H_gate(ancilla)

    if qem_type == "FB":
        qc.add_gate(Measurement(ancilla, 0))
        qc.add_CNOT_gate(ancilla, system)
    elif qem_type == "detection":
        qc_fail = qc.copy()
        qc.add_gate(P0(ancilla))
        qc_fail.add_gate(P1(ancilla))


    ptm_tot = circuit_to_ptm(qc)
    ptm = take_ptrace_ptm(ptm_tot, [system])
    
    if qem_type == "detection" and normalize_ptm:
        ptm /= ptm[0, 0]
        return ptm
    elif qem_type == "detection" and not normalize_ptm:
        ptm_fail = take_ptrace_ptm(circuit_to_ptm(qc_fail), [system])
        return ptm, ptm_fail
    else:
        return ptm

    #error_ptm_qem_array.append(ptm)    

def _get_X_corrected_ptm(ptm, qem_type = "FB", normalize_ptm = True):
    nbits = ptm.shape[0].bit_length() - 1
    assert  nbits%2 ==0, f"ptm dim = {2**nbits} must be powers of 4. maybe the input is unitary?"
    n_qubit = (ptm.shape[0].bit_length() - 1)//2
    assert n_qubit == 1
    system, ancilla = [0, 1]    
    
    qc1 = QuantumCircuit(n_qubit + 1) # 1 for ancilla
    qc2 = QuantumCircuit(n_qubit + 1) # 1 for ancilla

    qc1.add_H_gate(ancilla)
    qc1.add_CZ_gate(ancilla, system)

    #qc.add_dense_matrix_gate(range(n_qubit), unitary)

    qc2.add_CZ_gate(ancilla, system)
    qc2.add_H_gate(ancilla)

    if qem_type == "FB":
        qc2.add_gate(Measurement(ancilla, 0))
        qc2.add_CNOT_gate(ancilla, system)
    elif qem_type == "detection":
        qc2_fail = qc2.copy()
        qc2.add_gate(P0(ancilla))
        qc2_fail.add_gate(P1(ancilla))


    ptm1 = circuit_to_ptm(qc1)
    ptm2 = circuit_to_ptm(qc2)
    iden = np.diag(np.ones(2**1))

    ptm_tot = ptm2 @ np.kron(unitary_to_ptm(iden),ptm) @ ptm1 # original
    #ptm_tot = ptm2 @ np.kron(ptm, unitary_to_ptm(iden)) @ ptm1
    

    ptm_ret = take_ptrace_ptm(ptm_tot, [system])
    if qem_type == "detection" and normalize_ptm:
        ptm_ret /= ptm_ret[0, 0]
        return ptm_ret
    elif qem_type == "detection" and not normalize_ptm:
        ptm_fail = circuit_to_ptm(qc2_fail) @ np.kron(unitary_to_ptm(iden),ptm) @ ptm1
        ptm_fail = take_ptrace_ptm(ptm_fail, [system])
        return ptm_ret, ptm_fail
    else:
        return ptm

    #error_ptm_qem_array.append(ptm)

def get_Z_twirled_ptm(ptm):
    
    # unitaries
    s = np.diag([1, 1j])
    z = np.diag([1, -1])
    sdag = s.conj().T
    
    # ptms
    ptm_iden = unitary_to_ptm(np.diag([1, 1]))
    ptm_s = unitary_to_ptm(s)
    ptm_z = unitary_to_ptm(z)
    ptm_sdag = unitary_to_ptm(s@s@s)    
    ptm_Zsdag = unitary_to_ptm(sdag @ z)
    ptm_sZ = unitary_to_ptm(z @ s)
    
    ptm_1a = ptm_iden
    ptm_1b = ptm_iden
    
    ptm_2a = ptm_z
    ptm_2b = ptm_z
    
    ptm_3a = ptm_sdag
    ptm_3b = ptm_s
    
    ptm_4a = ptm_Zsdag
    ptm_4b = ptm_sZ

    twirl_set = [(ptm_1a, ptm_1b), (ptm_2a, ptm_2b), (ptm_3a, ptm_3b), (ptm_4a, ptm_4b)]
    return np.mean([ptm_b @ ptm @ ptm_a for (ptm_a, ptm_b) in twirl_set], axis = 0)    