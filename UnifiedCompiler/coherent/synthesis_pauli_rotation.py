#from compilation import _run_gridsynth_eps, _run_gridsynth_bits, add_circuit
from concurrent.futures import ThreadPoolExecutor

from UnifiedCompiler.coherent.solovay_kitaev import (
    word_to_gate,
    _run_gridsynth_bits, 
    _run_gridsynth_eps,
    conjugate_gateword, 
    get_conjugated_gateword, 
    get_gateword
)
from UnifiedCompiler.utils.circuit_utils import circuit_to_unitary

import subprocess
from subprocess import PIPE

from qiskit import QuantumCircuit

####################################
# Main functions
####################################

def add_circuit(qc_main, qc_added):
    for i in range(qc_added.get_gate_count()):
        qc_main.add_gate(qc_added.get_gate(i))

def synthesize_single_pauli_rotation_in_parallel(angles, gate_type, eps=None, bits = None, max_threads = 8, backend="qulacs", as_matrix = False):
    """returns unitary circuit that is Ross-Selinger approximation     
     attributes:
         angle: rotation angle of Pauli Rotation
         bits : integer, precision in number of bits
         eps : float, absolute precision in operator norm
         digits : integer, precision in decimal
    """
    # ThreadPoolExecutor を使用して並列処理を実行します。
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # タスクを executor に渡し、結果を取得します。
        
        if eps is not None: 
            word_list = list(executor.map(_run_gridsynth_eps, angles, [eps] * len(angles)))    
        
        if bits is not None: 
            word_list = list(executor.map(_run_gridsynth_bits, angles, [bits] * len(angles)))    
            
    circuit_tmp_list = [word_to_gate(_word, backend) for _word in word_list]

    circuit_list = []
    idx = 0
    for _ in range(len(angles)):
        if backend == "qiskit":
            from qiskit import QuantumCircuit
            circuit = QuantumCircuit(1)
            if gate_type == "Rz":
                circuit = circuit.compose(circuit_tmp_list[idx])
            elif gate_type == "Rx":
                circuit.h(0)
                circuit = circuit.compose(circuit_tmp_list[idx])
                circuit.h(0)
            elif gate_type == "Ry":
                circuit.sdg(0)
                circuit.h(0)
                circuit = circuit.compose(circuit_tmp_list[idx])
                circuit.h(0)
                circuit.s(0)
        elif backend == "qulacs":
            from qulacs import QuantumCircuit as QC                
            circuit = QC(1)
            if gate_type == "Rz":
                add_circuit(circuit, circuit_tmp_list[idx])                
            elif gate_type == "Rx":
                circuit.add_H_gate(0)
                add_circuit(circuit, circuit_tmp_list[idx])
                circuit.add_H_gate(0)                
            elif gate_type == "Ry":
                circuit.add_Sdag_gate(0)
                circuit.add_H_gate(0)
                add_circuit(circuit, circuit_tmp_list[idx])
                circuit.add_H_gate(0)
                circuit.add_S_gate(0)                
            
        circuit_list.append(circuit)
        idx += 1        
    if as_matrix:
        return [circuit_to_unitary(_qc, ) for _qc in circuit_list]
    
    return circuit_list    
    
def synthesize_single_pauli_rotation_conj_in_parallel(angles, gate_type, eps=None, bits = None, max_threads = 8, backend="qulacs", as_matrix = False):
    """returns unitary circuit that is Ross-Selinger approximation     
     attributes:
         angle: rotation angle of Pauli Rotation
         bits : integer, precision in number of bits
         eps : float, absolute precision in operator norm
         digits : integer, precision in decimal
    """
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        
        if eps is not None: 
            word_list = list(executor.map(_run_gridsynth_eps, angles, [eps] * len(angles)))    
        
        if bits is not None: 
            word_list = list(executor.map(_run_gridsynth_bits, angles, [bits] * len(angles)))    
            
    word_list = [conjugate_gateword(_gw) for _gw in word_list]
    circuit_tmp_list = [word_to_gate(_word, backend) for _word in word_list]

    circuit_list = []
    idx = 0
    for _ in range(len(angles)):
        if backend == "qiskit":
            from qiskit import QuantumCircuit
            circuit = QuantumCircuit(1)
            if gate_type == "Rz":
                circuit.h(0)
                circuit.s(0)
                ##
                circuit.s(0)
                circuit.h(0)
                circuit = circuit.compose(circuit_tmp_list[idx])
                circuit.h(0)
                circuit.sdg(0)
                ##   
                circuit.sdg(0)
                circuit.h(0)
            elif gate_type == "Rx":
                circuit.s(0)
                ## Ry
                circuit.s(0)
                circuit.h(0)
                circuit = circuit.compose(circuit_tmp_list[idx])
                circuit.h(0)
                circuit.sdg(0)
                ## Ry
                circuit.sdg(0)
            elif gate_type == "Ry":
                circuit.s(0)
                circuit.h(0)
                circuit = circuit.compose(circuit_tmp_list[idx])
                circuit.h(0)
                circuit.sdg(0)
        elif backend == "qulacs":
            from qulacs import QuantumCircuit as QC                
            circuit = QC(1)
            if gate_type == "Rz":
                circuit.add_H_gate(0)
                circuit.add_S_gate(0)
                ##
                circuit.add_S_gate(0)
                circuit.add_H_gate(0)
                add_circuit(circuit, circuit_tmp_list[idx])
                circuit.add_H_gate(0)
                circuit.add_Sdag_gate(0)                
                ## 
                circuit.add_Sdag_gate(0)
                circuit.add_H_gate(0)
            elif gate_type == "Rx":
                circuit.add_S_gate(0)
                ##
                circuit.add_S_gate(0)
                circuit.add_H_gate(0)
                add_circuit(circuit, circuit_tmp_list[idx])
                circuit.add_H_gate(0)
                circuit.add_Sdag_gate(0)                
                ## 
                circuit.add_Sdag_gate(0)
            elif gate_type == "Ry":
                ##
                circuit.add_S_gate(0)
                circuit.add_H_gate(0)
                add_circuit(circuit, circuit_tmp_list[idx])
                circuit.add_H_gate(0)
                circuit.add_Sdag_gate(0)                
                ## 
            
        circuit_list.append(circuit)
        idx += 1        
    if as_matrix:
        return [circuit_to_unitary(_qc, ) for _qc in circuit_list]
    
    return circuit_list      

####################################################################################
# Individual synthesis
####################################################################################

def synthesize_Rz_gate(angle, bits= None, eps = None, digits = None, backend = "qiskit", as_matrix = False):
    
    if bits is not None:
        proc = subprocess.run(f"gridsynth {angle} -b {bits}", shell=True, stdout=PIPE, stderr=PIPE, text=True)
    if eps is not None:
        proc = subprocess.run(f"gridsynth {angle} -e {eps}", shell=True, stdout=PIPE, stderr=PIPE, text=True)
    if digits is not None:
        proc = subprocess.run(f"gridsynth {angle} -d {digits}", shell=True, stdout=PIPE, stderr=PIPE, text=True)
    
    gateword = proc.stdout
    gateword = get_gateword(angle, bits= bits, eps = eps, digits = digits, )

    if backend == "qiskit":
        circuit = QuantumCircuit(1)
        circuit = circuit.compose(word_to_gate(gateword, backend = backend))
    elif backend == "qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        add_circuit(circuit, word_to_gate(gateword, backend=backend))

    if as_matrix:
        return circuit_to_unitary(circuit)
    return circuit

def synthesize_Rx_gate(angle, bits= None, eps = None, digits = None, backend = "qiskit", as_matrix = False):

    if backend == "qiskit":
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit = circuit.compose(synthesize_Rz_gate(angle, bits=bits, eps=eps, digits=digits, backend = backend))
        circuit.h(0)
    elif backend == "qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        circuit.add_H_gate(0)
        add_circuit(circuit, synthesize_Rz_gate(angle, bits=bits, eps=eps, digits=digits, backend = backend))
        circuit.add_H_gate(0)
    if as_matrix:
        return circuit_to_unitary(circuit)
    return circuit

def synthesize_Ry_gate(angle, bits= None, eps = None, digits = None, backend = "qiskit", as_matrix = False):
    if backend == "qiskit":
        circuit = QuantumCircuit(1)
        circuit.sdg(0)
        circuit.h(0)
        circuit = circuit.compose(synthesize_Rz_gate(angle, bits=bits, eps=eps, digits=digits, backend = backend))
        circuit.h(0)
        circuit.s(0)
    elif backend == "qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        circuit.add_Sdag_gate(0)
        circuit.add_H_gate(0)
        add_circuit(circuit, synthesize_Rz_gate(angle, bits=bits, eps=eps, digits=digits, backend = backend))
        circuit.add_H_gate(0)
        circuit.add_S_gate(0)
    if as_matrix:
        return circuit_to_unitary(circuit)
    return circuit

def synthesize_single_qubit_gate(angle, gate_type, bits= None, eps = None, digits = None, backend = "qiskit", as_matrix = False):
    if gate_type == "Rz":
        return synthesize_Rz_gate(angle, bits=bits, eps=eps, digits=digits, backend=backend, as_matrix = as_matrix)
    elif gate_type == "Rx":
        return synthesize_Rx_gate(angle, bits=bits, eps=eps, digits=digits, backend=backend, as_matrix = as_matrix)
    elif gate_type == "Ry":
        return synthesize_Ry_gate(angle, bits=bits, eps=eps, digits=digits, backend=backend, as_matrix = as_matrix)    
    
def synthesize_single_qubit_gate_conj(angle, gate_type, bits= None, eps = None, digits = None, backend = "qiskit", as_matrix = False):
    if gate_type == "Rz":
        return synthesize_Rz_gate_conj(angle, bits=bits, eps=eps, digits=digits, backend=backend, as_matrix = as_matrix)
    elif gate_type == "Rx":
        return synthesize_Rx_gate_conj(angle, bits=bits, eps=eps, digits=digits, backend=backend, as_matrix = as_matrix)
    elif gate_type == "Ry":
        return synthesize_Ry_gate_conj(angle, bits=bits, eps=eps, digits=digits, backend=backend, as_matrix = as_matrix)
    


def synthesize_Rx_gate_conj(angle, bits= None, eps = None, digits = None, backend="qiskit", as_matrix = False):
    ry_circ = synthesize_Ry_gate_conj(angle, bits=bits, eps=eps, digits=digits, backend=backend)
    
    if backend == "qiskit":
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit = circuit.compose(ry_circ)
        circuit.sdg(0)
    elif backend == "qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        circuit.add_S_gate(0)
        add_circuit(circuit, ry_circ)
        circuit.add_Sdag_gate(0)
    if as_matrix:
        return circuit_to_unitary(circuit)

    return circuit    

def synthesize_Rz_gate_conj(angle, bits= None, eps = None, digits = None, backend="qiskit", as_matrix = False):
    ry_circ = synthesize_Ry_gate_conj(angle, bits=bits, eps=eps, digits=digits, backend=backend)
    if backend == "qiskit":
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(1)
        
        circuit.h(0)
        circuit.s(0)        
        circuit = circuit.compose(ry_circ)
        circuit.sdg(0)        
        circuit.h(0)
        
    elif backend == "qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        circuit.add_H_gate(0)
        circuit.add_S_gate(0)        
        add_circuit(circuit, ry_circ)
        circuit.add_Sdag_gate(0)
        circuit.add_H_gate(0)
        
    if as_matrix:
        return circuit_to_unitary(circuit)

    return circuit    

def synthesize_Ry_gate_conj(angle, bits= None, eps = None, digits = None, backend="qiskit", as_matrix = False):
    gateword_conj = get_conjugated_gateword(angle, bits=bits, eps=eps, digits=digits)
    rz_conj = word_to_gate(gateword_conj, backend = backend)
    
    if backend == "qiskit":
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.h(0)
        circuit = circuit.compose(rz_conj)
        circuit.h(0)
        circuit.sdg(0)
    elif backend == "qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        circuit.add_S_gate(0)
        circuit.add_H_gate(0)
        add_circuit(circuit, rz_conj)
        circuit.add_H_gate(0)
        circuit.add_Sdag_gate(0)
    if as_matrix:
        return circuit_to_unitary(circuit)

    return circuit    

def single_qubit_gate(angle, gate_type, backend = "qiskit", as_matrix = False, ):
    if backend=="qiskit":
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(1)
        if gate_type == "Rz":
            circuit.rz(angle, 0)
        if gate_type == "Rx":
            circuit.rx(angle, 0)
        if gate_type == "Ry":
            circuit.ry(angle, 0)
    elif backend=="qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        if gate_type == "Rz":
            circuit.add_RZ_gate(0, -angle)
        if gate_type == "Rx":
            circuit.add_RX_gate(0, -angle)
        if gate_type == "Ry":
            circuit.add_RY_gate(0, -angle)        
    if as_matrix:
        return circuit_to_unitary(circuit)
    return circuit  
