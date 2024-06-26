

from UnifiedCompiler.probabilistic.with_scipy import (
    _solve_ptm_scipy, 
    solve_diagonal_ptm, 
    solve_depol_ptm, 
    solve_XY_ptm,
    solve_X_ptm,
    solve_Z_ptm,
    solve_XY_nondiag_ptm
    )

from UnifiedCompiler.probabilistic.with_gurobi import (
    solve_ptm_gurobi,
    _solve_ptm_gurobi_from_choi
)

PTM_SOLVER_TYPE = ["scipy", "gurobi", "cplex", "scipy-choi", "gurobi-choi"]
CONSTRAINT_TYPE = ["pauli", 
                   "depol", 
                   "XY", 
                   "X", 
                   "Z", 
                   "XY-eq", 
                   "XY-nondiag", 
                   "depol-ineq",
                   "Xnew",
                   "XYnew",
                   "none"
                   ]
def solve_ptm(
        error_ptm_list, 
        solver_type = "scipy", 
        constraint = "pauli",
        atol_min = 1e-17, 
        btol_min = 1e-17,
        method = "highs",
        verbose = 0,
        return_res = False,
        presolve = -1,
        tcounts = None,
        tcount_coeff = 0.0,
        dnorm_list = None,
        dnorm_coeff = 0.0
        ):
    """
    Compute probability mixture to best-approximate identity channel in terms of PTM,
    such that final output yields Pauli error (i.e. non-diagonal part is zero up to `atol`).

    attributes:
        error_ptm_list: list of PTM for U_exact^\dag @ U_compiled
    """    
    assert solver_type in PTM_SOLVER_TYPE, f"choose from {PTM_SOLVER_TYPE}"
    assert constraint in CONSTRAINT_TYPE, f"choose from {CONSTRAINT_TYPE}"

    if "choi" in solver_type:
        return solve_ptm_from_choi(
            error_ptm_list, 
            solver_type = solver_type, 
            constraint = constraint, 
            atol_min = atol_min, 
            btol_min = btol_min, 
            method = method, 
            presolve = presolve, 
            return_res = return_res,
            tcounts = tcounts,
            tcount_coeff = tcount_coeff,
            dnorm_list = dnorm_list,
            dnorm_coeff = dnorm_coeff
            )
    
    if solver_type == "scipy":
        if tcounts is not None and constraint not in ["X", "pauli", "none"]:
            raise NotImplementedError()
        if dnorm_list is not None and constraint not in ["none", "pauli"]:
            raise NotImplementedError()
        
        if constraint == "none":    
            return _solve_ptm_scipy(error_ptm_list, verbose = verbose, tcounts = tcounts, tcount_coeff = tcount_coeff, dnorm_list = dnorm_list , dnorm_coeff = dnorm_coeff)
        elif constraint == "pauli":
            sol, _atol, res = solve_diagonal_ptm(error_ptm_list, atol_min = atol_min, return_tol = True, method = method, verbose = verbose, return_res = True, tcounts = tcounts, tcount_coeff = tcount_coeff)
            #return sol
        elif constraint == "XY":
            sol, _atol, _btol, res = solve_XY_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose,return_res = True)
            #return sol        
        elif constraint == "XY-eq":
            sol, _atol, _btol, res = solve_XY_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, diagonal_constraint_as_equality=True, return_res = True)
            #return sol        
        elif constraint == "XY-nondiag":
            sol, _atol, _btol, res = solve_XY_nondiag_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, return_res = True)
            #return sol                    
        
        elif constraint == "X":
            sol, _atol, _btol, res = solve_X_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, return_res = True, tcounts = tcounts, tcount_coeff = tcount_coeff)

        # X new relaxes to allow coherent X rotation
        elif constraint == "Xnew":
            sol, _atol, _btol, res = solve_X_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, return_res = True, constraint = constraint)
            #return sol        

        # XY new relaxes to allow nondiagonal XY components
        elif constraint == "XYnew":
            sol, _atol, _btol, res = solve_XY_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, return_res = True, constraint = constraint)
            #return sol                    

            #return sol        
        elif constraint == "Z":
            sol, _atol, _btol, res = solve_Z_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, return_res = True)
            #return sol        
        
        elif constraint == "depol":
            sol, _atol, _btol, res = solve_depol_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, return_res = True)
        elif constraint == "depol-ineq":            
            sol, _atol, _btol, res = solve_depol_ptm(error_ptm_list, atol_min=atol_min, btol_min = btol_min, return_tol = True, method = method, verbose = verbose, return_res = True, diagonal_constraint_as_equality = False)
            #return sol
        else:
            raise NotImplementedError()
        
        if return_res:
            return sol, res
        return sol
    
    elif solver_type == "gurobi":
        from UnifiedCompiler.probabilistic.with_gurobi import _solve_ptm_gurobi
        return solve_ptm_gurobi(error_ptm_list, constraint = constraint, atol_min = atol_min, btol_min = btol_min, return_res = return_res)

    elif solver_type == "cplex":
        from UnifiedCompiler.probabilistic.with_cplex import _solve_ptm_cplex
        return _solve_ptm_cplex(error_ptm_list, constraint)
    else:
        raise NotImplementedError(f"{solver_type=} not implemented")
    
from UnifiedCompiler.utils.channel_utils import ptm_to_mb_vec
from UnifiedCompiler.probabilistic.with_gurobi import _solve_ptm_gurobi_from_choi
from UnifiedCompiler.probabilistic.with_scipy import _solve_ptm_from_choi_scipy
from scipy.optimize import linprog
import numpy as np
import copy
def solve_ptm_from_choi(
        error_ptm_list, 
        solver_type = "scipy", 
        constraint ="pauli", 
        method = "highs",
        atol_min = 1e-17,
        btol_min = 1e-17, 
        verbose = 0, 
        return_res = False, 
        return_tol = False,
        presolve = 0,
        tcounts = None,
        tcount_coeff = 0.0,
        dnorm_list = None,
        dnorm_coeff = 0.0
        ):
    assert solver_type in ["scipy", "gurobi", "scipy-choi", "gurobi-choi"], f"{solver_type=} not implemented. "
    #assert constraint in "CONST"

#def solve_ptm_pauli_from_choi(error_ptm_list, atol_min = 1e-17, btol_min = 1e-17, verbose = 0, return_res = False, return_tol = False):
    _atol = copy.deepcopy(atol_min)
    _btol = copy.deepcopy(btol_min)
    solution = None
    error_mbvec_list = np.array([ptm_to_mb_vec(_ptm) for _ptm in error_ptm_list])
    error_mbmat_list = np.array([np.outer(_mbvec, _mbvec.conj()).reshape(len(_mbvec)**2) for _mbvec in error_mbvec_list])    
    
    #constraint = "pauli"
    
    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm_from_choi(
                error_mbmat_list, 
                solver_type = solver_type, 
                atol = _atol, 
                btol = _btol, 
                constraint = constraint, 
                method = method, 
                return_res = True, 
                presolve = presolve,
                tcounts = tcounts,
                tcount_coeff = tcount_coeff,
                dnorm_list = dnorm_list,
                dnorm_coeff = dnorm_coeff
                )
            
            if solution is None:
                if verbose > 1:
                    print(f"raising from _atol = {_atol:.2e}")
                _atol *= np.sqrt(10)
                _btol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm_from_choi(
            error_mbmat_list, 
            atol = _atol, 
            btol = _btol, 
            constraint = constraint, 
            method = method, 
            return_res = True,
            tcounts = tcounts,
            tcount_coeff = tcount_coeff,
            dnorm_list = dnorm_list,
            dnorm_coeff = dnorm_coeff            
            )
    
    ret = [solution]
    if return_tol:
        ret.append(_atol)
        ret.append(_btol)
    if return_res:
        ret.append(res)
        
    if len(ret)==1:
        return ret[0]
    return ret

def _solve_ptm_from_choi(
        error_mbmat_list, 
        solver_type = "scipy",
        atol = 1e-16, 
        btol = 1e-16, 
        constraint = "pauli", 
        method = "highs", 
        return_res = False,
        presolve = 0,
        tcounts = None,
        tcount_coeff = 0.0,
        dnorm_list = None,
        dnorm_coeff = 0.0        
        ):
    if solver_type in ["scipy", "scipy-choi"]:
        #if tcounts is not None:
            #raise NotImplementedError()
        return _solve_ptm_from_choi_scipy(
            error_mbmat_list, 
            atol = atol, 
            btol = btol, 
            constraint = constraint, 
            method = method, 
            return_res = return_res, 
            tcounts = tcounts,
            tcount_coeff=tcount_coeff,
            dnorm_list = dnorm_list,
            dnorm_coeff = dnorm_coeff            
        )
    elif solver_type in ["gurobi", "gurobi-choi"]:
        return _solve_ptm_gurobi_from_choi(
            error_mbmat_list, 
            atol = atol, 
            btol = btol, 
            constraint = constraint, 
            return_res = return_res,
            presolve = presolve,
            tcounts = tcounts,
            tcount_coeff = tcount_coeff,
            dnorm_list = dnorm_list,
            dnorm_coeff = dnorm_coeff
            )


def test_mb_choi():
    from qulacs import QuantumCircuit
    from qulacs.gate import Probabilistic, X, Y, Z
    from UnifiedCompiler.utils.channel_utils import circuit_to_ptm, ptm_to_choi_MB
    qc = QuantumCircuit(1)
    qc.add_gate(Probabilistic([0.03], [X(0)]))

    ptm = circuit_to_ptm(qc)
    print("magic basis choi under X error:")
    print(np.round(ptm_to_choi_MB(ptm).real, 7),)
    
    qc = QuantumCircuit(1)
    qc.add_gate(Probabilistic([0.03], [Y(0)]))

    ptm = circuit_to_ptm(qc)
    print("\nmagic basis choi under Y error:")
    print(np.round(ptm_to_choi_MB(ptm).real, 7),)
    
    qc = QuantumCircuit(1)
    qc.add_gate(Probabilistic([0.03], [Z(0)]))

    ptm = circuit_to_ptm(qc)
    print("\nmagic basis choi under Z error:")
    print(np.round(ptm_to_choi_MB(ptm).real, 7),)    