import subprocess
from subprocess import PIPE
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from UnifiedCompiler.utils.circuit_utils import circuit_to_unitary

##################################
# single-qubit compilation
#################################

def get_gateword(angle, bits= None, eps = None, digits = None, ):
    if bits is not None:
        proc = subprocess.run(f"gridsynth {angle} -b {bits}", shell=True, stdout=PIPE, stderr=PIPE, text=True)
    if eps is not None:
        proc = subprocess.run(f"gridsynth {angle} -e {eps}", shell=True, stdout=PIPE, stderr=PIPE, text=True)
    if digits is not None:
        proc = subprocess.run(f"gridsynth {angle} -d {digits}", shell=True, stdout=PIPE, stderr=PIPE, text=True)
    
    gateword = proc.stdout
    return gateword

def get_conjugated_gateword(angle, bits= None, eps = None, digits = None, ):
    return conjugate_gateword(get_gateword(angle, bits=bits, eps=eps,digits=digits))

def conjugate_gateword(gateword):
    gatelist = [s for s in gateword]
    gatelist_conj = []
    for g in gatelist:
        if g=="S":
            gatelist_conj.append("SSS")
        elif g=="T":
            gatelist_conj.append("SSST")
        else:
            gatelist_conj.append(g)
    gateword_conj = "".join(gatelist_conj)    
    return gateword_conj

def _run_gridsynth_bits(angle, bits):
    proc = subprocess.run(
        f"gridsynth {angle} -b {bits}",
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    return proc.stdout

def _run_gridsynth_eps(angle, eps):
    proc = subprocess.run(
        f"gridsynth {angle} -e {eps}",
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    return proc.stdout

def word_to_gate(gateword, backend = "qulacs"):
    phase = np.pi/4 * gateword.count("W")
    
    if backend == "qulacs":
        from qulacs import QuantumCircuit as QC
        circuit = QC(1)
        circuit.add_dense_matrix_gate(0, np.array([[np.exp(1j * phase), 0], [0, np.exp(1j * phase)]]))
        for g in gateword:
            if g=="H":
                circuit.add_H_gate(0)
            elif g=="T":
                #circuit.add_dense_matrix_gate(0, [[1, 0], [0, np.exp(1j * np.pi/4)]])
                circuit.add_T_gate(0)
            elif g=="S":
                #circuit.add_dense_matrix_gate(0, [[1, 0], [0, np.exp(1j * np.pi/2)]])
                circuit.add_S_gate(0)
            elif g=="X":
                circuit.add_X_gate(0)                
        
    elif backend == "qiskit":
        circuit = QuantumCircuit(1, global_phase = phase)
        for g in gateword:
            if g=="H":
                circuit.h(0)
            elif g == "T":
                circuit.t(0)
            elif g=="S":
                circuit.s(0)
            elif g=="X":
                circuit.x(0)
    return circuit

def word_to_unitary(gateword, backend = "qulacs"):
    assert backend in ["qulacs", "qiskit"]
    circuit = word_to_gate(gateword, backend)
    if backend == "qiskit":
        unitary = Operator(circuit).data
    elif backend == "qulacs":
        unitary = circuit_to_unitary(circuit, )

    return unitary


def count_t_gates(circuit):
    count = 0
    if "qiskit" in str(type(circuit)):
        dic = circuit.count_ops()
        
        try:
            count += dic["t"]
        except:
            None
            
        try:
            count += dic["tdg"]
        except:
            None
        
    elif "qulacs" in str(type(circuit)):   
        for i in range(circuit.get_gate_count()):
            g = circuit.get_gate(i)
            if g.get_name() == "T":
                count += 1
            elif g.get_name() == "Tdag":
                count += 1
    else:
        raise Exception(f"type = {type(circuit)} is not accepted.")
    return count