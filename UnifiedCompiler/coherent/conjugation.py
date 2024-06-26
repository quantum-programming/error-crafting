from UnifiedCompiler.coherent.solovay_kitaev import conjugate_gateword, word_to_gate
import numpy as np

##################################################
# Gate conjugation for pauli rotations
##################################################
from qulacs import QuantumCircuit as QC
from qiskit import QuantumCircuit

def add_circuit(qc_main, qc_added):
    for i in range(qc_added.get_gate_count()):
        qc_main.add_gate(qc_added.get_gate(i))

def conjugate_pauli_rotation(circuit, gate_type, ):
    if gate_type == "Ry":
        return conjugate_Ry_gate(circuit)
    elif gate_type == "Rz":
        return conjugate_Rz_gate(circuit)
    if gate_type == "Rx":
        return conjugate_Rx_gate(circuit)

def get_rz_gatewords_from_ry(circuit):

    if isinstance(circuit, QC):
        rz_gate_word_list = []
        for idx in range(circuit.get_gate_count())[3:-2]:
            if circuit.get_gate(idx).get_name() == "H":
                rz_gate_word_list.append("H")
            elif circuit.get_gate(idx).get_name() == "T":
                rz_gate_word_list.append("T")
            elif circuit.get_gate(idx).get_name() == "S":
                rz_gate_word_list.append("S")
            elif circuit.get_gate(idx).get_name() == "X":
                rz_gate_word_list.append("X")
            else:
                raise Exception(f"gate : {_gate.operation.name} not expected.")

        rz_gate_word = "".join(rz_gate_word_list)     
    else:
        from qiskit import QuantumCircuit
        if isinstance(circuit, QuantumCircuit):
            rz_gate_word_list = []
            for _gate in circuit[2:-2]:
                if _gate.operation.name == "h":
                    rz_gate_word_list.append("H")
                elif _gate.operation.name == "t":
                    rz_gate_word_list.append("T")
                elif _gate.operation.name == "s":
                    rz_gate_word_list.append("S")
                elif _gate.operation.name == "x":
                    rz_gate_word_list.append("X")
                else:
                    raise Exception(f"gate : {_gate.operation.name} not expected.")

            rz_gate_word = "".join(rz_gate_word_list)        
           
    return rz_gate_word    

def conjugate_Ry_gate(circuit):

    #rz_gate_word = "".join(rz_gate_word_list)
    rz_gate_word = get_rz_gatewords_from_ry(circuit)
    rz_gate_word_conj = conjugate_gateword(rz_gate_word)

    if isinstance(circuit, QC):
        rz_conj = word_to_gate(rz_gate_word_conj, backend = "qulacs")

        circuit_conj = QC(1)
        #circuit.add_dense_matrix_gate(0, np.array([[np.exp(1j * phase), 0], [0, np.exp(1j * phase)]]))
        #circuit_conj = QuantumCircuit(1, global_phase = np.conjugate(circuit.global_phase))
        circuit_conj.add_S_gate(0)
        circuit_conj.add_H_gate(0)
        add_circuit(circuit_conj, rz_conj)
        #circuit_conj = circuit_conj.compose(rz_conj)
        circuit_conj.add_H_gate(0)
        circuit_conj.add_Sdag_gate(0)    

        return circuit_conj.copy()    
    else:
        from qiskit import QuantumCircuit
        if isinstance(circuit, QuantumCircuit):
            rz_conj = word_to_gate(rz_gate_word_conj, backend = "qiskit")

            circuit_conj = QuantumCircuit(1, global_phase = np.conjugate(circuit.global_phase))
            circuit_conj.s(0)
            circuit_conj.h(0)
            circuit_conj = circuit_conj.compose(rz_conj)
            circuit_conj.h(0)
            circuit_conj.sdg(0)    
        
            return circuit_conj.copy()

        else:
            raise NotImplementedError()
    
def conjugate_Rz_gate(circuit):
    if isinstance(circuit, QuantumCircuit):
        ry_circuit = QuantumCircuit(1)
        ry_circuit.sdg(0)
        ry_circuit.h(0)
        ry_circuit = ry_circuit.compose(circuit)
        ry_circuit.h(0)
        ry_circuit.s(0)
        
        ry_circuit_conj = conjugate_Ry_gate(ry_circuit)
        rz_circuit_conj = QuantumCircuit(1)

        rz_circuit_conj.h(0)
        rz_circuit_conj.s(0)
        rz_circuit_conj = rz_circuit_conj.compose(ry_circuit_conj)
        rz_circuit_conj.sdg(0)
        rz_circuit_conj.h(0)
    
        return rz_circuit_conj.copy()  
    elif isinstance(circuit, QC):
        ry_circuit = QC(1)
        ry_circuit.add_Sdag_gate(0)
        ry_circuit.add_H_gate(0)
        add_circuit(ry_circuit, circuit)
        #ry_circuit = ry_circuit.compose(circuit)
        ry_circuit.add_H_gate(0)
        ry_circuit.add_S_gate(0)
        
        ry_circuit_conj = conjugate_Ry_gate(ry_circuit)
        rz_circuit_conj = QC(1)

        rz_circuit_conj.add_H_gate(0)
        rz_circuit_conj.add_S_gate(0)
        add_circuit(rz_circuit_conj, ry_circuit_conj)
        #rz_circuit_conj = rz_circuit_conj.compose(ry_circuit_conj)
        rz_circuit_conj.add_Sdag_gate(0)
        rz_circuit_conj.add_H_gate(0)

        return rz_circuit_conj.copy()
    else:
        raise NotImplementedError()
      
    
def conjugate_Rx_gate(circuit):
    if isinstance(circuit, QuantumCircuit):
        ry_circuit = QuantumCircuit(1)
        ry_circuit.sdg(0)
        ry_circuit = ry_circuit.compose(circuit)
        ry_circuit.s(0)
        
        ry_circuit_conj = conjugate_Ry_gate(ry_circuit)
        rx_circuit_conj = QuantumCircuit(1)

        rx_circuit_conj.s(0)
        rx_circuit_conj = rx_circuit_conj.compose(ry_circuit_conj)
        rx_circuit_conj.sdg(0)
    
        return rx_circuit_conj.copy()     
    elif isinstance(circuit, QC):
        ry_circuit = QC(1)
        ry_circuit.add_Sdag_gate(0)
        add_circuit(ry_circuit, circuit)
        #ry_circuit = ry_circuit.compose(circuit)
        ry_circuit.add_S_gate(0)
        
        ry_circuit_conj = conjugate_Ry_gate(ry_circuit)
        rx_circuit_conj = QC(1)

        rx_circuit_conj.add_S_gate(0)
        add_circuit(rx_circuit_conj, ry_circuit_conj)
        #rx_circuit_conj = rx_circuit_conj.compose(ry_circuit_conj)
        rx_circuit_conj.add_Sdag_gate(0)
    
        return rx_circuit_conj.copy()     
    else:
        raise NotImplementedError()
       

#####################################
# four-fold conjugation
######################################

def add_circuit(qc_main, qc_added):
    for i in range(qc_added.get_gate_count()):
        qc_main.add_gate(qc_added.get_gate(i))

def four_fold_conjugation(circuit, gate_type):
    if gate_type != "Rz":
        raise NotImplementedError()
        
    if type(circuit) == QC: #qulacs
        n_qubit = circuit.get_qubit_count()
        
        circuit1 = circuit.copy()
        circuit2 = QC(n_qubit)
        circuit3 = QC(n_qubit)
        circuit4 = QC(n_qubit)
        
        circuit2.add_Z_gate(0)
        add_circuit(circuit2, circuit.copy())
        circuit2.add_Z_gate(0)
        
        circuit3.add_Sdag_gate(0)
        add_circuit(circuit3, circuit.copy())
        circuit3.add_S_gate(0)        
        
        circuit4.add_Z_gate(0)
        circuit4.add_Sdag_gate(0)        
        add_circuit(circuit4, circuit.copy())
        circuit4.add_S_gate(0)
        circuit4.add_Z_gate(0)
        
        
        qc_list = [circuit1, circuit2, circuit3, circuit4]
        
    elif type(circuit) == QuantumCircuit: #qiskit
        
        n_qubit = circuit.num_qubits
        circuit1 = circuit.copy()

        
        circuit4 = QuantumCircuit(n_qubit)
        
        circuit2 = QuantumCircuit(n_qubit)
        circuit2.z(0)
        circuit2.compose(circuit.copy(), inplace = True)
        circuit2.z(0)
        
        circuit3 = QuantumCircuit(n_qubit)
        circuit3.sdg(0)
        circuit3.compose(circuit.copy(), inplace = True)
        circuit3.s(0)        
        
        
        circuit4 = QuantumCircuit(n_qubit)
        circuit4.z(0)        
        circuit4.sdg(0)
        circuit4.compose(circuit.copy(), inplace = true)
        circuit4.s(0)        
        circuit4.z(0)
 
        
        qc_list = [circuit1, circuit2, circuit3, circuit4]
        
    return qc_list