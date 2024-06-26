from qulacs import ParametricQuantumCircuit
from qulacs import QuantumCircuit as QC
from qulacs.gate import CZ, RY,RZ, merge, ParametricRY, ParametricRZ

#from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge

####################################
# Quantum Circuits
####################################
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from typing import Union
from scipy.sparse import csc_matrix, identity
from qulacs.circuit import QuantumCircuitOptimizer as QCO
from qulacs.gate import merge

def circuit_to_unitary(circuit):
    
    # qulacs
    if type(circuit) == QC:
        n_qubit = circuit.get_qubit_count()
        qco = QCO()
        circuit_ = circuit.copy()
        qco.optimize_light(circuit_)
        for ind in range(3, n_qubit+1):
            qco.optimize(circuit_, ind)
        gate_list = [circuit_.get_gate(i) for i in range(circuit_.get_gate_count())]
        merged = merge(gate_list)
        return merged.get_matrix()
    
    # qiskit
    elif type(circuit) == QuantumCircuit:
        return Operator(circuit).data
    else:
        raise NotImplementedError()

from qulacs.gate import DenseMatrix
def PQC_to_QC(circuit:ParametricQuantumCircuit):
    n_qubit = circuit.get_qubit_count()
    circuit_new = QC(n_qubit)
    for i in range(circuit.get_gate_count()):
        gate = circuit.get_gate(i)
        assert gate.get_name() in ["ParametricPauliRotation", "ParametricRX", "ParametricRY", "ParametricRZ"]

        ind_list = gate.get_target_index_list()
        mat = gate.get_matrix()
        gate_new = DenseMatrix(index_list = ind_list, matrix = mat)
        circuit_new.add_gate(gate_new)

    return circuit_new

def circuit_to_matrix(circuit:Union[QC, ParametricQuantumCircuit], ):
    n_qubit = circuit.get_qubit_count()
    qco = QCO()
    if isinstance(circuit, ParametricQuantumCircuit):
        circuit_ = PQC_to_QC(circuit) 
    elif isinstance(circuit, QC):
        circuit_ = circuit.copy()
    qco.optimize_light(circuit_)
    for ind in range(3, n_qubit+1):
        qco.optimize(circuit_, ind)
    gate_list = [circuit_.get_gate(i) for i in range(circuit_.get_gate_count())]
    merged = merge(gate_list)
    return merged.get_matrix()

def inverse_unitary(qc:QC):
    qc_inv = QC(qc.get_qubit_count())
    for i in range(qc.get_gate_count())[::-1]:
        qc_inv.add_gate(qc.get_gate(i).get_inverse())    
    return qc_inv

def inverse_unitary_old(n_qubit, depth, theta_list, PBC=False):
    # TODO: create Udag for arbitrary parametrized circuit

    circuit = QC(n_qubit)
    if depth != 0:
        for i in range(n_qubit):
            circuit.add_gate(merge(RZ(i, -theta_list[2*i+1+2*n_qubit*depth]), RY(i, -theta_list[2*i+2*n_qubit*depth])))
                
    for d in reversed(range(depth-1)):
        if PBC:
            circuit.add_gate(CZ(n_qubit - 1, 0))
        
        for i in range(n_qubit//2-1 + n_qubit%2):
            circuit.add_gate(CZ(2*i+1, 2*i+2))
            
        for i in range(n_qubit//2):
            circuit.add_gate(CZ(2*i, 2*i+1))
        
        for i in range(n_qubit):
            circuit.add_gate(merge(RZ(i, -theta_list[2*i+1+2*n_qubit*d]), RY(i, -theta_list[2*i+2*n_qubit*d])))
            
    return circuit        