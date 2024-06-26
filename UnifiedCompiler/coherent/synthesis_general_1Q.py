from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import numpy as np

from UnifiedCompiler.utils.circuit_utils import circuit_to_unitary
from UnifiedCompiler.coherent.synthesis_pauli_rotation import synthesize_Rx_gate, synthesize_Rz_gate
from UnifiedCompiler.coherent.solovay_kitaev import _run_gridsynth_bits, _run_gridsynth_eps, word_to_gate
from UnifiedCompiler.coherent.shift_unitary import MBU, shiftnormalizedr, shiftnormalizedr_depol

from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qulacs import QuantumCircuit as QC

def get_u3_angles(unitary):
    assert unitary.shape[0] == 2, "not applicable to n>1 qubits now."
    oed = OneQubitEulerDecomposer(basis = "ZXZ")
    
    angles_oed = oed.angles(unitary)
    angles_oed = [angles_oed[2]%(2*np.pi), angles_oed[0] % (2*np.pi), angles_oed[1]%(2*np.pi)]
    return angles_oed

def unitary_to_gate(unitary):
    return exact_u_gate(*get_u3_angles(unitary))

def exact_u_gate(angle1, angle2, angle3, as_matrix = False, backend = "qiskit"):
    """returns unitary circuit that is
     -- RZ(angle1) -- RX(angle2) -- RZ(angle3)
    """
    if backend=="qiskit":
        circuit = QuantumCircuit(1)
        circuit.rz(angle1, 0)
        circuit.rx(angle2, 0)
        circuit.rz(angle3, 0)
        
        if as_matrix:
            return Operator(circuit).data
        return circuit
    elif backend == "qulacs":
        circuit = QC(1)
        circuit.add_RZ_gate(0, -angle1)
        circuit.add_RX_gate(0, -angle2)
        circuit.add_RZ_gate(0, -angle3)

        if as_matrix:
            from UnifiedCompiler.utils.circuit_utils import circuit_to_unitary
            return circuit_to_unitary(circuit)
        return circuit

def add_circuit(qc_main, qc_added):
    for i in range(qc_added.get_gate_count()):
        qc_main.add_gate(qc_added.get_gate(i))
        
##################################################################
# Main functions
##################################################################

#from UnifiedCompiler.coherent.direct_search import _run_direct_search

def synthesize_u_gate_direct(angles, eps, backend = "qulacs", as_matrix = False):
    from UnifiedCompiler.coherent.direct_search import _run_direct_search
    gateword = _run_direct_search(U, eps, )    
    
    circuit = word_to_gate(gateword, backend)
    
    if as_matrix:
        return circuit_to_unitary(circuit)
    return circuit

def synthesize_u_gates_in_parallel_direct(angles_list, eps, backend = "qulacs", as_matrix = False, max_threads = 8):
    from UnifiedCompiler.coherent.direct_search import _run_direct_search
    with ThreadPoolExecutor(max_workers = max_threads) as executor:
        word_list = list(executor.map(_run_direct_search, angles_list, [eps,] * len(angles_list)))
    
    circuit_list = [word_to_gate(_word, backend) for _word in word_list]
    
    if as_matrix:
        return [circuit_to_unitary(_circuit) for _circuit in circuit_list]
    return circuit_list

def synthesize_u_gates_in_parallel(
    angles_list: List[List], 
    bits = None, 
    eps = None, 
    digits = None, 
    as_matrix= False,
    max_threads = 8,
    backend = "qulacs"
):
    """returns unitary circuit that is Ross-Selinger approximation of
     -- RZ(angle1) -- RX(angle2) -- RZ(angle3)
     
     attributes:
         angle1: rotation angle of first RZ
         angle2: rotation angle of RX
         angle3: rotation angle of second RZ
         bits : integer, precision in number of bits
         eps : float, absolute precision in operator norm
         digits : integer, precision in decimal
    """
    assert all([len(_a)==3 for _a in angles_list])
    angles = sum(angles_list, [])
    #rotation_basis_list = sum([["Z", "X", "Z"] for _ in range(len(angles_list))], [])
    
    # ThreadPoolExecutor を使用して並列処理を実行します。
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # タスクを executor に渡し、結果を取得します。
        
        if eps is not None:
            #print(f"{isinstance(eps, float)=}")
            if isinstance(eps, float) or isinstance(eps, np.float64): 
                word_list = list(executor.map(_run_gridsynth_eps, angles, [eps] * len(angles)))    
            elif type(eps) in [np.ndarray, list]:
                eps_ = list(eps)
                assert len(eps_) == 3
                word_list = list(executor.map(_run_gridsynth_eps, angles, eps_ * (len(angles)//3)))
            else:
                raise Exception(f"type(eps) = {type(eps)} unknown")
        
        if bits is not None: 
            word_list = list(executor.map(_run_gridsynth_bits, angles, [bits] * len(angles)))    
            
    circuit_tmp_list = [word_to_gate(_word, backend) for _word in word_list]

    circuit_list = []
    idx = 0
    for _ in range(len(angles_list)):
        if backend == "qiskit":
            qc = QuantumCircuit(1)
            qc.compose(circuit_tmp_list[idx], inplace = True)
            qc.h(0)
            qc.compose(circuit_tmp_list[idx+1], inplace = True)
            qc.h(0)
            qc.compose(circuit_tmp_list[idx+2], inplace = True)
        elif backend == "qulacs":
            from qulacs import QuantumCircuit as QC
            qc = QC(1)
            add_circuit(qc, circuit_tmp_list[idx])
            qc.add_H_gate(0)
            add_circuit(qc, circuit_tmp_list[idx+1])
            qc.add_H_gate(0)
            add_circuit(qc, circuit_tmp_list[idx+2])
        else:
            raise Exception("backend unknown")
            
        circuit_list.append(qc)
        idx += 3        
    #for 
    if as_matrix:
        return [circuit_to_unitary(_qc, ) for _qc in circuit_list]
    
    return circuit_list


def synthesize_u_gate(angle1, angle2, angle3, bits = None, eps = None, digits = None, as_matrix= False, verbose = 0):
    """returns unitary circuit that is Ross-Selinger approximation of
     -- RZ(angle1) -- RX(angle2) -- RZ(angle3)
     
     attributes:
         angle1: rotation angle of first RZ
         angle2: rotation angle of RX
         angle3: rotation angle of second RZ
         bits : integer, precision in number of bits
         eps : float, absolute precision in operator norm
         digits : integer, precision in decimal
    """
    
    #from qiskit import QuantumCircuit
    import qiskit
    circuit = qiskit.QuantumCircuit(1)
    if verbose:
        print(circuit)
    circuit = circuit.compose(synthesize_Rz_gate(angle1, bits, eps, digits, backend = "qiskit"), )
    circuit = circuit.compose(synthesize_Rx_gate(angle2, bits, eps, digits, backend = "qiskit"), )
    circuit = circuit.compose(synthesize_Rz_gate(angle3, bits, eps, digits, backend = "qiskit"), )
    if as_matrix:
        return Operator(circuit).data
    
    return circuit

######################################################
# Epsilon-net shifts
######################################################

def generate_epsilon_net_unitaries(u_target:np.ndarray, eps:float, c:float, error_type = "pauli", n_shift_unitary = None):
    if n_shift_unitary is not None:
        shifting_unitary_list = make_single_qubit_shiftunitary(c, eps, dim = n_shift_unitary)
    else:
        assert error_type in ["pauli", "depol", "XY", "X", "depol-ineq"], "Choose error type from pauli, depol and XY."
        if error_type == "pauli":
            shifting_unitary_list = [magic_basis_choi_to_unitary(np.array([np.sqrt(1-(c*eps)**2), c*eps , c*eps, c*eps])* _r)  for _r in shiftnormalizedr]
        elif error_type in ["depol", "XY", "X", "depol-ineq"]:
            shifting_unitary_list = [magic_basis_choi_to_unitary(np.array([np.sqrt(1-(c*eps)**2), c*eps , c*eps, c*eps])* _r)  for _r in shiftnormalizedr_depol]
        else:
            raise Exception()
    return [_u @ u_target for _u in shifting_unitary_list]

def generate_epsilon_net_circuits(u_target, eps, c, return_unitary_list = False, error_type = "pauli", n_shift_unitary = None):
    # Todo : check consistency of epsilon

    # generate epsilon net by Akibue-san's method
    u_ideal_net_list = generate_epsilon_net_unitaries(u_target, eps, c, error_type = error_type, n_shift_unitary = n_shift_unitary)
    angles_list = [get_u3_angles(_u) for _u in u_ideal_net_list]
    
    # Ross-Selinger compilation
    qc_approx_list = [synthesize_u_gate(*_angles, eps = eps,) for _angles in angles_list]
    u_compiled_net_list = [Operator(_qc).data for _qc in qc_approx_list]
    if return_unitary_list:
        return qc_approx_list, u_compiled_net_list
    return qc_approx_list


def canonical_to_magic_basis_choi_matrix(choi_matrix):
    n_qubit = int(np.log2(choi_matrix.shape[0])//2)
    if n_qubit > 1:
        raise NotImplementedError
    return MBU @ choi_matrix @ MBU.conj().T

def magic_basis_choi_to_unitary(magic_choivec):
    # Todo: asssert it is unitary
    n_qubit = int(np.log2(magic_choivec.shape[0])/2)
    if n_qubit > 1:
        raise NotImplementedError
    choivec = MBU.conj().T @ magic_choivec
    return np.sqrt(2) * choivec.reshape(2**n_qubit, 2**n_qubit).T

def make_single_qubit_shiftunitary(c, eps, dim = 7):
    import os, json
    shift_unitaries_dir = './UnifiedCompiler/coherent/shift_normalized_r'
    file_path = os.path.join(shift_unitaries_dir, f'dim_{dim}.data')

    if not os.path.exists(shift_unitaries_dir):
        os.makedirs(shift_unitaries_dir)    

    if os.path.exists(file_path):
        # Read the shift_normalized_r from the files
        shift_normalized_r = json.load(open(file_path, "r"))
        #with open(file_path, 'r') as file:
            #shift_normalized_r = [list(map(float, line.strip().split())) for line in file]        

    else:
        r_list_tmp = get_repulsive_points(dim).tolist()
        shift_normalized_r = [[1] + _r for _r in r_list_tmp]
        
        json.dump(shift_normalized_r, open(file_path, "w"))
        #with open(file_path, 'w') as file:
            #for item in shift_normalized_r:
                #file.write(' '.join(map(str, item)) + '\n')        
    return [magic_basis_choi_to_unitary(np.array([np.sqrt(1-(c*eps)**2), c*eps , c*eps, c*eps])* _r)  for _r in shift_normalized_r] 


def get_repulsive_points(dim):
    if dim > 3000:
        raise Exception("check the runtime to generate shift unitary")

    # Initialize points on the sphere
    points = np.random.randn(dim, 3)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    lr = 0.01  # learning rate
    for _ in range(300):
        # Vectorized computation of pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1) + np.eye(dim)  # Avoid division by zero
        
        # Compute repulsion force, avoiding division by zero
        force = diff / dist[..., np.newaxis]**3
        net_force = np.sum(force - force.transpose((1, 0, 2)), axis=1)/2 # divided by 2 for accordance with old
        
        # Update points
        points -= lr * net_force
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    return points

def get_repulsive_points_old(dim):
    import numpy as np
    if dim > 500:
        raise Exception("check the runtime to generate shift unitary")

    N = dim

    # initialize
    points = np.random.randn(N, 3)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    def repulsion_force(points):
        force = np.zeros((N, 3))
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    f = diff / dist**3
                    force[i] += f
                    force[j] -= f
        return force

    # gradient descent
    lr = 0.01  # learning rate
    for _ in range(300):
        force = repulsion_force(points)
        points -= lr * force
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    return points
