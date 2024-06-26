

def test_ptrace_ptm():
    from UnifiedCompiler.utils import circuit_to_unitary, unitary_to_ptm
    from UnifiedCompiler.utils.ptrace_utils import take_ptrace_state, take_ptrace_ptm
    from qulacs.gate import Identity
    from qulacs.gate import RandomUnitary, to_matrix_gate
    import numpy as np
    from qulacs import QuantumCircuit
    #import numpy
    
    qc = QuantumCircuit(3)
    u0, u1, u2 = [to_matrix_gate(RandomUnitary([0], seed=seed)).get_matrix() for i, seed in enumerate([123, 1234,12345])]
    qc.add_dense_matrix_gate(0, u0, )
    qc.add_dense_matrix_gate(1, u1, )
    qc.add_dense_matrix_gate(2, u2, )

    u = circuit_to_unitary(qc)

    ptm = unitary_to_ptm(u)    
    
    
    # case 1
    sites = sorted([0, 1])
    ptmm = take_ptrace_ptm(ptm, sites)
    myqc = QuantumCircuit(len(sites))
    if 0 in sites:
        #myqc.add_Z_gate(sites.index(0))
        #myqc.add_gate(u0, sites.index(0))
        myqc.add_dense_matrix_gate(sites.index(0), u0)
    if 1 in sites:
        #myqc.add_X_gate(sites.index(1))
        #myqc.add_gate(u1, sites.index(1))
        myqc.add_dense_matrix_gate(sites.index(1), u1)
    if 2 in sites:
        #myqc.add_Y_gate(sites.index(2))
        #myqc.add_gate(u2, sites.index(2))
        myqc.add_dense_matrix_gate(sites.index(2), u2)

    u_ = circuit_to_unitary(myqc)
    ptm2 = unitary_to_ptm(u_)
    
    print(f"sites = {sites} is consistent?", np.allclose(ptm2, ptmm))
    
    # case 2
    sites = sorted([1, 2])
    ptmm = take_ptrace_ptm(ptm, sites)
    myqc = QuantumCircuit(len(sites))
    if 0 in sites:
        #myqc.add_Z_gate(sites.index(0))
        #myqc.add_gate(u0, sites.index(0))
        myqc.add_dense_matrix_gate(sites.index(0), u0)
    if 1 in sites:
        #myqc.add_X_gate(sites.index(1))
        #myqc.add_gate(u1, sites.index(1))
        myqc.add_dense_matrix_gate(sites.index(1), u1)
    if 2 in sites:
        #myqc.add_Y_gate(sites.index(2))
        #myqc.add_gate(u2, sites.index(2))
        myqc.add_dense_matrix_gate(sites.index(2), u2)
    u_ = circuit_to_unitary(myqc)
    ptm2 = unitary_to_ptm(u_)
    
    print(f"sites = {sites} is consistent?", np.allclose(ptm2, ptmm))
    
    # case 3
    sites = sorted([0, 2])
    ptmm = take_ptrace_ptm(ptm, sites)
    myqc = QuantumCircuit(len(sites))
    if 0 in sites:
        #myqc.add_Z_gate(sites.index(0))
        #myqc.add_gate(u0, sites.index(0))
        myqc.add_dense_matrix_gate(sites.index(0), u0)
    if 1 in sites:
        #myqc.add_X_gate(sites.index(1))
        #myqc.add_gate(u1, sites.index(1))
        myqc.add_dense_matrix_gate(sites.index(1), u1)
    if 2 in sites:
        #myqc.add_Y_gate(sites.index(2))
        #myqc.add_gate(u2, sites.index(2))
        myqc.add_dense_matrix_gate(sites.index(2), u2)

    u_ = circuit_to_unitary(myqc)
    ptm2 = unitary_to_ptm(u_)
    
    print(f"sites = {sites} is consistent?", np.allclose(ptm2, ptmm))    

def test_PTM_calculation_consistency(n_qubit = 1, seed1=123, seed2 = 12345):
    import qutip as qt
    import numpy as np
    from UnifiedCompiler.utils import state_in_pauli_basis, circuit_to_ptm
    from qulacs import DensityMatrix, QuantumCircuit
    
        
    # prepare random unitaries
    u = qt.rand_unitary_haar(2**n_qubit, seed=seed1).full()
    v = qt.rand_unitary_haar(2**n_qubit, seed=seed2).full()
    
    # initial state
    vec0 = np.array([1,] + [0 for _ in range(2**n_qubit - 1)])
    vec0 = u @ vec0
    rho0 = np.outer(vec0, vec0.conj())
    rhovec0 = state_in_pauli_basis(vec0)    
    
    ########################
    # matrix-based calculation
    ########################
    rho = v @ rho0 @ v.conj().T
    rhovec1 = state_in_pauli_basis(rho)
    
    ########################
    # circuit based calculation
    ########################
    state = DensityMatrix(n_qubit)
    prep = QuantumCircuit(n_qubit)
    qc = QuantumCircuit(n_qubit)

    prep.add_dense_matrix_gate(range(n_qubit), u)
    prep.update_quantum_state(state)
    
    qc.add_dense_matrix_gate(range(n_qubit), v)
    qc.update_quantum_state(state)
    rho2 = state.get_matrix()
    rhovec2 = state_in_pauli_basis(rho2)
    
    ########################
    # PTM-based calculation
    ########################
    ptm = circuit_to_ptm(qc)
    rhovec3 = ptm @ rhovec0
    
    print(f"PTM-based state calculations all consistent? {np.allclose(rhovec1, rhovec2) and np.allclose(rhovec2, rhovec3)}")
    

def test_feedback_PTM_calculation_consistency(n_qubit = 1, seed1=123, seed2 = 12345):
    import qutip as qt
    from qulacs import QuantumCircuit, DensityMatrix
    from qulacs.gate import Measurement
    import numpy as np
    from UnifiedCompiler.utils import state_in_pauli_basis, circuit_to_ptm
    from UnifiedCompiler.utils.ptrace_utils import take_ptrace_state, take_ptrace_ptm
    

    assert n_qubit == 1, "not implemented."
    # prepare random unitaries
    u = qt.rand_unitary_haar(2**n_qubit, seed=seed1).full()
    v = qt.rand_unitary_haar(2**n_qubit, seed=seed2).full()
    ancilla = n_qubit
    system = 0

    # feedback circuit
    qc = QuantumCircuit(n_qubit + 1) # 1 for ancilla
    qc.add_H_gate(ancilla)
    qc.add_CZ_gate(ancilla, system)
    qc.add_dense_matrix_gate(range(n_qubit), v)
    qc.add_CZ_gate(ancilla, system)
    qc.add_H_gate(ancilla)

    qc.add_gate(Measurement(ancilla, 0))
    qc.add_CNOT_gate(ancilla, system)

    # initial state
    vec0 = np.array([1,] + [0 for _ in range(2**n_qubit - 1)])
    vec0 = u @ vec0
    rho0 = np.outer(vec0, vec0.conj())
    rhovec0 = state_in_pauli_basis(vec0)    

    ########################
    # circuit based calculation
    ########################
    state = DensityMatrix(n_qubit+1)
    prep = QuantumCircuit(n_qubit+1)

    prep.add_dense_matrix_gate(range(n_qubit), u)
    prep.update_quantum_state(state)

    qc.update_quantum_state(state)
    rho2 = take_ptrace_state(state.get_matrix(), range(n_qubit))
    rhovec2 = state_in_pauli_basis(rho2)

    ########################
    # PTM-based calculation
    ########################
    #ptm = circuit_to_ptm(qc)
    ptm = take_ptrace_ptm(circuit_to_ptm(qc), list(range(n_qubit)))
    rhovec3 = ptm @ rhovec0

    print(f"PTM calculations for feedback circuits consistent? {np.allclose(rhovec2, rhovec3)}")    


def test_ptrace_action():
    import numpy as np
    from UnifiedCompiler.utils import unitary_to_ptm
    from UnifiedCompiler.utils.noise_utils import MyDepolarizingNoise
    import scipy
    from qulacs import QuantumCircuit
    from UnifiedCompiler.utils import circuit_to_ptm
    from UnifiedCompiler.utils.channel_utils import get_X_corrected_ptm

    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    u_rx = scipy.linalg.expm(1j * 0.3 * sigmax)
    ptm_rx = unitary_to_ptm(u_rx)
    print(f"\n---------------------------")
    print(f"rx ptm: ")
    print(ptm_rx.real)

    print(f"\nafter X correction : ")
    print(get_X_corrected_ptm(ptm_rx).real)

    u_ry = scipy.linalg.expm(1j * 0.3 * sigmay)
    ptm_ry = unitary_to_ptm(u_ry)
    print(f"\n---------------------------")
    print(f"\nry ptm: ")
    print(ptm_ry.real)

    print(f"\nafter X correction : ")
    print(get_X_corrected_ptm(ptm_ry).real)


    u_rz = scipy.linalg.expm(1j * 0.25 * sigmaz)
    ptm_rz = unitary_to_ptm(u_rz)
    print(f"\n---------------------------")
    print(f"rz ptm: ")
    print(ptm_rz.real)

    print(f"\nafter X correction : ")
    print(get_X_corrected_ptm(ptm_rz).real)


    qc = QuantumCircuit(1)
    qc.add_gate(MyDepolarizingNoise(0, 0.03))
    ptm_dep = circuit_to_ptm(qc)
    print(f"\n---------------------------")
    print(f"\ndepol ptm: ")
    print(ptm_dep.real)

    print(f"\nafter X correction : ")
    print(get_X_corrected_ptm(ptm_dep).real)


if __name__ == "__main__":
    test_ptrace_ptm()
    test_PTM_calculation_consistency()
    test_feedback_PTM_calculation_consistency()
    test_ptrace_action()
