import numpy as np


from UnifiedCompiler.utils import state_in_pauli_basis

# References:
# https://qutip.org/docs/latest/guide/guide-random.html
# https://qiskit.org/documentation/_modules/qiskit/quantum_info/states/random.html#random_density_matrix


def _dm_check(dm: np.ndarray):
    # From the Born Rule, the following constraints are necessary
    # * Tr(dm) = 1
    # * dm ⪰ 0
    # * dm = dm^†
    assert dm.ndim == 2
    assert dm.shape[0] == dm.shape[1] and bin(dm.shape[0]).count("1") == 1
    assert np.isclose(np.trace(dm), 1.0)
    assert np.all(np.linalg.eigvals(dm) + 1e-10 >= 0), np.linalg.eigvals(dm)
    assert np.allclose(np.transpose(dm), np.conj(dm))
    print("ok!")


def _make_random_mixed_density_matrix(n_qubit: int, seed: int) -> np.ndarray:
    import qutip
    return qutip.rand_dm_ginibre(2**n_qubit, seed=seed).full()


def _make_random_pure_density_matrix(n_qubit: int, seed: int) -> np.ndarray:
    import qutip
    return qutip.ket2dm(qutip.rand_ket_haar(2**n_qubit, seed=seed)).full()



def make_random_quantum_state(n_qubit: int, kind: str, seed: int=1234, check: bool = False):
    """
    kind: ["mixed", "pure"]
    n_qubit : integer
    seed = 
    """
    assert kind in ["mixed", "pure"], kind
    assert 0 <= seed < 10000, f"seed must be in [0, 10000), but {seed=}"
    
    if kind == "mixed":
        dm = _make_random_mixed_density_matrix(n_qubit, seed)
    else:
        dm = _make_random_pure_density_matrix(n_qubit, seed)
    if check:
        _dm_check(dm)
    return state_in_pauli_basis(dm, check=check)

import numpy as np
from scipy.stats import unitary_group
from qulacs.gate import RandomUnitary

##################################
# Full random matrices
#################################

def haar_random_unitary(num_qubit, seed = None, size = 1):
    if size !=1:
        return _haar_random_unitary_scipy(num_qubit, seed = seed , size = size)
    else:
        if num_qubit < 7:
            return _haar_random_unitary_qulacs(num_qubit, seed = seed)
        else:
            return _haar_random_unitary_scipy(num_qubit, seed = seed)

def _haar_random_unitary_qulacs(num_qubit, seed = None):
    if seed is not None:
        return RandomUnitary(list(range(num_qubit)), seed = seed).get_matrix()        
    else:
        return RandomUnitary(list(range(num_qubit)), ).get_matrix()        

def _haar_random_unitary_scipy(num_qubit, seed = None, size = 1):
    return unitary_group(2**num_qubit, seed = seed).rvs(size)


def main():
    for n in range(1, 5 + 1):
        print(f"{n=}")
        for seed in range(5):
            print(f"{seed=}")
            print(make_random_quantum_state("mixed", n, seed, check=True)[:5])
            print(make_random_quantum_state("pure", n, seed, check=True)[:5])

    for n in range(6, 11 + 1):
        print(f"{n=}")
        print(make_random_quantum_state("mixed", n, 0, check=False)[:5])
        print(make_random_quantum_state("pure", n, 0, check=False)[:5])


if __name__ == "__main__":
    main()
