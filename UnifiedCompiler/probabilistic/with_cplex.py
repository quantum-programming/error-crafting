import scipy
import numpy as np
from UnifiedCompiler.utils.channel_utils import unitary_to_ptm
import itertools

def to_LP_problem(error_ptm_list, use_sparse_csr = True):

    n_qubit = int(np.log2(error_ptm_list[0].shape[0])/2)
    A = np.array([_ptm.reshape(_ptm.size) for _ptm in error_ptm_list]).T.real
    if use_sparse_csr:
        A = scipy.sparse.csr_matrix(A)
        
    b = unitary_to_ptm(np.identity(2**n_qubit)).reshape(16**n_qubit).real
    return A,b

def _solve_ptm_cplex(error_ptm_list, use_sparse_csr = True, scale = 1e3, constraint = "none"):    
    A, b = to_LP_problem(error_ptm_list, use_sparse_csr = use_sparse_csr)

    return __solve_ptm_cplex(A*scale, b * scale, constraint = constraint)

def __solve_ptm_cplex(A, b, constraint = "none"):
    from docplex.mp.model import Model
    m, n = A.shape
    assert len(b) == m, "Inconsistent dimensions"

    # Create a model
    mdl = Model(name='L1_norm_minimization')

    # Create variables
    x = mdl.continuous_var_list(n, lb = 0, name='x')
    e = mdl.continuous_var_list(m, lb = 0, name='e')

    # Create objective function
    mdl.minimize(mdl.sum(e))

    # Create constraints
    use_sparse_csr = isinstance(A, scipy.sparse.csr_matrix)
    if use_sparse_csr:
        #Ax = [mdl.scal_prod([x[j] for j in A.getrow(i).indices], A.getrow(i).data) for i in range(m)]
        Ax = []
        for i in range(m):
            #row_start = A.indptr[i]
            #row_end = A.indptr[i+1]
            row = A.getrow(i)
            row_values = row.data
            row_columns = row.indices
            Ax_i = mdl.scal_prod([x[j] for j in row_columns], row_values)
            Ax.append(Ax_i)

    else:
        # Create constraints
        Ax = [mdl.dot(x, A[i]) for i in range(m)]

    if constraint == "none":
        mdl.add_constraints(Ax[i] - b[i] <= e[i] for i in range(len(e)))
        mdl.add_constraints(Ax[i] - b[i] >=  - e[i] for i in range(len(e)))   
    elif constraint == "pauli":
        nondiag_args = [i *(4**i) + j for (i, j) in itertools.permutations(range(1, 4**n_qubit), 2)]
        mdl.add_constraints(Ax[i] - b[i] <= e[i] for i in range(m) if i not in nondiag_args)
        mdl.add_constraints(Ax[i] - b[i] >=  - e[i] for i in range(m) if i not in nondiag_args)
        mdl.add_constraints(Ax[i] == b[i] for i in range(m) if i in nondiag_args)

    mdl.add_constraint(mdl.sum(x) == 1)

    # Solve
    solution = mdl.solve()

    return [solution[_xi] for _xi in x]