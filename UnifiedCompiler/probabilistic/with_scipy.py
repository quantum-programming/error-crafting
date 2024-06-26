import numpy as np
from UnifiedCompiler.utils.channel_utils import unitary_to_ptm
from scipy.optimize import linprog

import itertools
import copy

CONSTRAINTS = ["none", "pauli", "depol", "depol-ineq", "XY", "X", "Z", "XY-eq", "XY-nondiag", "Xnew", "XYnew"]

def _solve_ptm_scipy(
        error_ptm_list, 
        verbose = 0, 
        tcounts = None, 
        tcount_coeff = 0.0, 
        dnorm_list = None, 
        dnorm_coeff = 0.0
        ):
    n_qubit = int(np.log2(error_ptm_list[0].shape[0])/2)
    A = np.array([_ptm.reshape(_ptm.size) for _ptm in error_ptm_list]).T.real
    b = unitary_to_ptm(np.identity(2**n_qubit)).reshape(16**n_qubit)
    
    n, m = A.shape
    
    # 目的関数の係数: (A x - b の L1ノルムを最小化するために)
    # (m + n)次元の係数を作成する
    c = np.concatenate((np.zeros(m), np.ones(n)))

    # minimize T-count
    if tcounts is not None:
        c += np.concatenate((np.array(tcounts) * tcount_coeff, np.zeros(n)))

    # maximize unitary synthesis error
    if dnorm_list is not None:
        c += np.concatenate((-np.array(dnorm_list) * dnorm_coeff, np.zeros(n)))

    # 制約条件: xの要素の和は1
    # A_eq: (1, 1, ..., 1, 0, 0, ..., 0)
    A_eq = np.concatenate((np.ones((1, m)), np.zeros((1, n))), axis=1)
    b_eq = np.array([1])

    # A x - b の L1ノルムを最小化
    A_ub = np.concatenate((A, -np.eye(n)), axis=1)
    A_ub = np.vstack((A_ub, np.concatenate((-A, -np.eye(n)), axis=1)))
    b_ub = np.concatenate((b, -b))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))

    if res.success:
        x = res.x[:m]
        return x
    else:
        return None
    
def solve_diagonal_ptm(
        error_ptm_list, 
        atol_min = 1e-16,
        return_tol = False, 
        method = "highs", 
        verbose = 0,
        return_res = False,
        presolve = 1,
        tcounts = None,
        tcount_coeff = 0.0,
        dnorm_list = None,
        dnorm_coeff = 0.0
        ):
    _atol = copy.deepcopy(atol_min)
    solution = None
    
    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm(
                error_ptm_list, 
                atol = _atol, 
                method = method,
                constraint = "pauli",
                return_res = True,
                presolve = presolve,
                tcounts = tcounts,
                tcount_coeff = tcount_coeff,
                dnorm_list = dnorm_list,
                dnorm_coeff = dnorm_coeff
                )
            
            if solution is None:
                _atol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm(
            error_ptm_list, 
            atol = _atol, 
            method = method,
            constraint = "pauli",
            return_res = True,
            presolve = presolve,
            tcounts = tcounts,
            tcount_coeff = tcount_coeff,
            dnorm_list = dnorm_list,
            dnorm_coeff = dnorm_coeff
        )
    ret = [solution]
    if return_tol:
        ret.append(_atol)
        #ret.append(_btol)
        #return solution, _atol, _btol
    if return_res:
        ret.append(res)
    if len(ret)==1:
        return ret[0]
    return ret

def solve_depol_ptm(
        error_ptm_list, 
        atol_min = 1e-17, 
        btol_min = 1e-17, 
        return_tol = False, 
        method = "highs", 
        diagonal_constraint_as_equality = True,
        verbose = 0,
        return_res = False,
        presolve = 1
    ):
    
    if diagonal_constraint_as_equality:
        constraint = "depol"
    else:
        constraint = "depol-ineq"

    _atol = copy.deepcopy(atol_min)
    _btol = copy.deepcopy(btol_min)
    solution = None
    
    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm(
                error_ptm_list, 
                atol = _atol, 
                btol = _btol,
                method = method,
                constraint = constraint,
                return_res = True,
                presolve = presolve
                )
            
            if solution is None:
                if verbose > 1:
                    print(f"raising from _atol = {_atol:.2e}")
                _atol *= np.sqrt(10)
                _btol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm(
            error_ptm_list, 
            atol = _atol, 
            btol = _btol,
            method = method,
            constraint = constraint,
            return_res = True,
            presolve = presolve
        )
    ret = [solution]
    if return_tol:
        ret.append(_atol)
        ret.append(_btol)
        #return solution, _atol, _btol
    if return_res:
        ret.append(res)
    if len(ret)==1:
        return ret[0]
    return ret

def solve_XY_ptm(
        error_ptm_list, 
        atol_min = 1e-17, 
        btol_min = 1e-17, 
        return_tol = False, 
        method = "highs", 
        verbose = 0,
        diagonal_constraint_as_equality = False,
        return_res = False,
        presolve = 1,
        constraint = "XY"
    ):
    
    if not diagonal_constraint_as_equality:
        constraint = "XY"
    else:
        constraint = "XY-eq"

    _atol = copy.deepcopy(atol_min)
    _btol = copy.deepcopy(btol_min)
    solution = None
    
    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm(
                error_ptm_list, 
                atol = _atol, 
                btol = _btol,
                method = method,
                constraint = constraint,
                return_res = True,
                presolve = presolve
                )
            
            if solution is None:
                if verbose > 1:
                    print(f"raising from _atol = {_atol:.2e}")
                _atol *= np.sqrt(10)
                _btol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm(
            error_ptm_list, 
            atol = _atol, 
            btol = _btol,
            method = method,
            constraint = constraint,
            return_res = True,
            presolve = presolve
        )
    ret = [solution]
    if return_tol:
        ret.append(_atol)
        ret.append(_btol)
        #return solution, _atol, _btol
    if return_res:
        ret.append(res)
    if len(ret)==1:
        return ret[0]
    return ret

def solve_XY_nondiag_ptm(
        error_ptm_list, 
        atol_min = 1e-17, 
        btol_min = 1e-17, 
        return_tol = False, 
        method = "highs", 
        verbose = 0,
        return_res = False,
        presolve = 1
    ):
    
    #if not diagonal_constraint_as_equality:
    constraint = "XY-nondiag"
    #else:
        #constraint = "XY-eq"

    _atol = copy.deepcopy(atol_min)
    _btol = copy.deepcopy(btol_min)
    solution = None
    
    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm(
                error_ptm_list, 
                atol = _atol, 
                btol = _btol,
                method = method,
                constraint = constraint,
                return_res = True,
                presolve = presolve
                )
            
            if solution is None:
                if verbose > 1:
                    print(f"raising from _atol = {_atol:.2e}")
                _atol *= np.sqrt(10)
                _btol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm(
            error_ptm_list, 
            atol = _atol, 
            btol = _btol,
            method = method,
            constraint = constraint,
            return_res = True,
            presolve = presolve
        )
    ret = [solution]
    if return_tol:
        ret.append(_atol)
        ret.append(_btol)
        #return solution, _atol, _btol
    if return_res:
        ret.append(res)
    if len(ret)==1:
        return ret[0]
    return ret

def solve_X_ptm(
        error_ptm_list, 
        atol_min = 1e-17, 
        btol_min = 1e-17, 
        return_tol = False, 
        method = "highs", 
        verbose = 0,
        return_res = False,
        presolve = 1,
        constraint = "X",
        tcounts = None,
        tcount_coeff = 0.01
    ):
    
    #constraint = "X"

    _atol = copy.deepcopy(atol_min)
    _btol = copy.deepcopy(btol_min)
    solution = None
    
    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm(
                error_ptm_list, 
                atol = _atol, 
                btol = _btol,
                method = method,
                constraint = constraint,
                return_res = True,
                presolve = presolve,
                tcounts = tcounts,
                tcount_coeff = tcount_coeff
                )
            
            if solution is None:
                if verbose > 1:
                    print(f"raising from _atol = {_atol:.2e}")
                _atol *= np.sqrt(10)
                _btol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm(
            error_ptm_list, 
            atol = _atol, 
            btol = _btol,
            method = method,
            constraint = constraint,
            return_res = True,
            presolve = presolve,
            tcounts = tcounts,
            tcount_coeff = tcount_coeff            
        )
    ret = [solution]
    if return_tol:
        ret.append(_atol)
        ret.append(_btol)
        #return solution, _atol, _btol
    if return_res:
        ret.append(res)
    if len(ret)==1:
        return ret[0]
    return ret

def solve_Z_ptm(
        error_ptm_list, 
        atol_min = 1e-17, 
        btol_min = 1e-17, 
        return_tol = False, 
        method = "highs", 
        verbose = 0,
        return_res = False,
        presolve = 1
    ):
    
    constraint = "Z"

    _atol = copy.deepcopy(atol_min)
    _btol = copy.deepcopy(btol_min)
    solution = None
    
    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm(
                error_ptm_list, 
                atol = _atol, 
                btol = _btol,
                method = method,
                constraint = constraint,
                return_res = True,
                presolve = presolve
                )
            
            if solution is None:
                if verbose > 1:
                    print(f"raising from _atol = {_atol:.2e}")
                _atol *= np.sqrt(10)
                _btol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm(
            error_ptm_list, 
            atol = _atol, 
            btol = _btol,
            method = method,
            constraint = constraint,
            return_res = True,
            presolve = presolve
        )
    ret = [solution]
    if return_tol:
        ret.append(_atol)
        ret.append(_btol)
        #return solution, _atol, _btol
    if return_res:
        ret.append(res)
    if len(ret)==1:
        return ret[0]
    return ret

def _solve_ptm(
        error_ptm_list, 
        atol = 1e-16, 
        btol = 1e-16, 
        method = "highs", 
        constraint = "pauli", 
        return_res = False,
        presolve = 1,
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
        atol : Controls the allowed error in nondiagonal elements of PTM. default : 1e-8            
    """

    assert constraint in CONSTRAINTS
    n_qubit = int(np.log2(error_ptm_list[0].shape[0])/2)
    A = np.array([_ptm.reshape(_ptm.size) for _ptm in error_ptm_list]).T.real
    b = unitary_to_ptm(np.identity(2**n_qubit)).reshape(16**n_qubit).real
    
    n, m = A.shape
    
    # 目的関数の係数: (A x - b の L1ノルムを最小化するために)
    # (m + n)次元の係数を作成する
    c = np.concatenate((np.zeros(m), np.ones(n)))
    if tcounts is not None:
        c += np.concatenate((np.array(tcounts) * tcount_coeff, np.zeros(n)))
    # maximize unitary synthesis error
    if dnorm_list is not None:
        c += np.concatenate((-np.array(dnorm_list) * dnorm_coeff, np.zeros(n)))        

    # 制約条件: xの要素の和は1
    # A_eq: (1, 1, ..., 1, 0, 0, ..., 0)
    A_eq = np.concatenate((np.ones((1, m)), np.zeros((1, n))), axis=1)
    b_eq = np.array([1], dtype = float)

    # A x - b の L1ノルムを最小化
    A_ub = np.concatenate((A, -np.eye(n)), axis=1)
    A_ub = np.vstack((A_ub, np.concatenate((-A, -np.eye(n)), axis=1)))
    b_ub = np.concatenate((b, -b))


    # 非対角要素に関する制約 (error = pauli, depol)
    #nondiag_args = [(1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (3, 2)]
    nondiag_args = [(i, j) for (i, j) in itertools.permutations(range(1, 4**n_qubit), 2)]
    A_ub_ = np.array([(A[arg[0]*(4**n_qubit) + arg[1]]).tolist() + [0 for _ in range(n)] for arg in nondiag_args])
    A_ub_ = np.vstack((A_ub_, -A_ub_))
    b_ub_ = [atol for _ in range(len(nondiag_args)*2)]

    if constraint == "pauli":
        A_ub = np.vstack((A_ub, A_ub_))
        b_ub = np.concatenate((b_ub, b_ub_))
    elif constraint == "XY":

        # 対角要素に関する制約 (as inequality)
        A_ub2 = np.array([
            (A[1*4 + 1] + A[2*4 + 2] - A[3*4 + 3]).tolist() + [0 for _ in range(n)],
        ])
        #(A[1*4 + 1] - A[2*4 + 2]).tolist() + [0 for _ in range(n)],

        A_ub2 = np.vstack((A_ub2, -A_ub2))
        #b_ub2 = [1 + btol, -1 + btol]    
        b_ub2 = [1 + btol, -1 + btol]    

        A_ub = np.vstack((A_ub, A_ub_, A_ub2))
        b_ub = np.concatenate((b_ub, b_ub_, b_ub2))        


    elif constraint == "X":
        
        # 対角要素に関する制約 (as inequality)
        #A_ub2 = np.array([
            #(A[2*4 + 2] - A[3*4 + 3]).tolist() + [0 for _ in range(n)],
        #])
        
        A_ub2 = np.array([(A[1*4**n_qubit + 1]).tolist() + [0 for _ in range(n)]])
        A_ub2 = np.vstack((A_ub2, -A_ub2))
        b_ub2 = [1+btol, -1+btol]    

        A_ub = np.vstack((A_ub, A_ub_, A_ub2,))
        b_ub = np.concatenate((b_ub, b_ub_, b_ub2))    

    elif constraint == "Xnew":
        
        # 対角要素に関する制約 (as inequality)
        #A_ub2 = np.array([
            #(A[2*4 + 2] - A[3*4 + 3]).tolist() + [0 for _ in range(n)],
        #])
        nondiag_args = [(i, j) for (i, j) in itertools.permutations(range(1, 4**n_qubit), 2) if (i,j) not in [(2, 3), (3, 2)]]
        A_ub_ = np.array([(A[arg[0]*(4**n_qubit) + arg[1]]).tolist() + [0 for _ in range(n)] for arg in nondiag_args])
        A_ub_ = np.vstack((A_ub_, -A_ub_))
        b_ub_ = [atol for _ in range(len(nondiag_args)*2)]        
        
        A_ub2 = np.array([(A[1*4**n_qubit + 1]).tolist() + [0 for _ in range(n)]])
        A_ub2 = np.vstack((A_ub2, -A_ub2))
        b_ub2 = [1+btol, -1+btol]    

        A_ub = np.vstack((A_ub, A_ub_, A_ub2,))
        b_ub = np.concatenate((b_ub, b_ub_, b_ub2))        

    elif constraint == "XYnew":
        
        # 対角要素に関する制約 (as inequality)
        #A_ub2 = np.array([
            #(A[2*4 + 2] - A[3*4 + 3]).tolist() + [0 for _ in range(n)],
        #])
        nondiag_args = [(i, j) for (i, j) in itertools.permutations(range(1, 4**n_qubit), 2) if (i,j) not in [(2, 3), (3, 2)]]
        A_ub_ = np.array([(A[arg[0]*(4**n_qubit) + arg[1]]).tolist() + [0 for _ in range(n)] for arg in nondiag_args])
        A_ub_ = np.vstack((A_ub_, -A_ub_))
        b_ub_ = [atol for _ in range(len(nondiag_args)*2)]        
        
        #A_ub2 = np.array([(A[1*4**n_qubit + 1]).tolist() + [0 for _ in range(n)]])
        A_ub2 = np.array([
            (A[1*4 + 1] + A[2*4 + 2] - A[3*4 + 3]).tolist() + [0 for _ in range(n)],
        ])        
        A_ub2 = np.vstack((A_ub2, -A_ub2))
        b_ub2 = [1+btol, -1+btol]    

        A_ub = np.vstack((A_ub, A_ub_, A_ub2,))
        b_ub = np.concatenate((b_ub, b_ub_, b_ub2))                             

    elif constraint == "XY-nondiag":
        raise Exception("this code is incorrect, use constraint = XY")

        # XY non-diags
        nondiag_args = [(1, 2), (2,1)]
        A_ub_ = np.array([(A[arg[0]*(4**n_qubit) + arg[1]]).tolist() + [0 for _ in range(n)] for arg in nondiag_args])
        A_ub_ = np.vstack((A_ub_, -A_ub_))
        b_ub_ = [atol for _ in range(len(nondiag_args)*2)]        
        A_ub = np.vstack((A_ub, A_ub_))
        b_ub = np.concatenate((b_ub, b_ub_))

        # ZX, ZY non-diags
        nondiag_args_Zonly = [(1, 3), (2, 3), (3,1), (3,2)]
        A_ub__ = np.array([(A[arg[0]*(4**n_qubit) + arg[1]]).tolist() + [0 for _ in range(n)] for arg in nondiag_args_Zonly])
        A_ub__ = np.vstack((A_ub__, -A_ub__))
        b_ub__ = [btol for _ in range(len(nondiag_args_Zonly)*2)]
        A_ub = np.vstack((A_ub, A_ub__))
        b_ub = np.concatenate((b_ub, b_ub__))
        
        #tol = 1e-5  # この値は適宜調整してください
        # 対角成分の[3, 3]要素が1である不等式制約
        A_ub_xy1 = np.array([(A[3*4**n_qubit + 3]).tolist() + [0 for _ in range(n)]])
        b_ub_xy1 = [1 + btol]
        A_ub_xy2 = -A_ub_xy1
        #b_ub_xy2 = [-1 + btol]
        b_ub_xy2 = [-1 + btol]

        A_ub = np.vstack((A_ub, A_ub_xy1, A_ub_xy2))
        b_ub = np.concatenate((b_ub, b_ub_xy1, b_ub_xy2))        

    elif constraint == "XY-eq":
        raise Exception("this code is incorrect, use constraint = XY.")
        # 非対角要素の制約は既に`A_ub_`と`b_ub_`に記述されているので利用
        #A_ub = np.vstack((A_ub, A_ub_))
        #b_ub = np.concatenate((b_ub, b_ub_))
        
        #tol = 1e-5  # この値は適宜調整してください
        # 対角成分の[3, 3]要素が1である不等式制約
        #A_ub_xy1 = np.array([(A[3*4**n_qubit + 3]).tolist() + [0 for _ in range(n)]])
        #b_ub_xy1 = [1 + btol]
        #A_ub_xy2 = -A_ub_xy1
        #b_ub_xy2 = [-1 + btol]
        #b_ub_xy2 = [-1 + btol]

        #A_ub = np.vstack((A_ub, A_ub_xy1, A_ub_xy2))
        #b_ub = np.concatenate((b_ub, b_ub_xy1, b_ub_xy2))        

        # 対角要素に関する制約 (error = depol)
        #A_eq2 = np.array([
        #    (A[3*4 + 3]*1e3).tolist() + [0 for _ in range(n)],
        #])
        #A_ub2 = np.vstack((A_eq2, -A_ub2))
        #b_eq2 = [0 for _ in range(1)]

        #A_eq = np.vstack((A_eq, A_eq2))
        #b_eq = np.concatenate((b_eq, b_eq2))        

    elif constraint == "depol-ineq":
        # 対角要素に関する制約 (as inequality)
        A_ub2 = np.array([
            (A[1*4 + 1] - A[2*4 + 2]).tolist() + [0 for _ in range(n)],
            (A[2*4 + 2] - A[3*4 + 3]).tolist() + [0 for _ in range(n)],
        ])
        A_ub2 = np.vstack((A_ub2, -A_ub2))
        b_ub2 = [btol for _ in range(2*2)]    

        A_ub = np.vstack((A_ub, A_ub_, A_ub2))
        b_ub = np.concatenate((b_ub, b_ub_, b_ub2))
        
    elif constraint == "depol":
        A_ub = np.vstack((A_ub, A_ub_))
        b_ub = np.concatenate((b_ub, b_ub_))        
    
        # 対角要素に関する制約 (error = depol)
        A_eq2 = np.array([
            (A[1*4 + 1] - A[2*4 + 2]).tolist() + [0 for _ in range(n)],
            (A[2*4 + 2] - A[3*4 + 3]).tolist() + [0 for _ in range(n)],
        ])
        #A_ub2 = np.vstack((A_eq2, -A_ub2))
        b_eq2 = [0 for _ in range(2)]

        A_eq = np.vstack((A_eq, A_eq2))
        b_eq = np.concatenate((b_eq, b_eq2))

    else:
        raise Exception(f"{constraint=} not implemeneted")

    # 1e-9 seems to be unchangable tolerance value in scipy
    if atol < 1e-9:
        mult = 1e-9/atol
        A_ub *= mult; b_ub *= mult; A_eq *= mult; b_eq *= mult; 

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method = method, options = {"maxiter":100, "presolve":presolve==1})

    try:
        x=res.x[:m]
        if return_res:
            return x/sum(x), res
        return x/sum(x)
    except:
        if return_res:
            return None, res
        return None
    #if res.success:
        #x = res.x[:m]
        #return x/sum(x)
    #else:
        #return None
    
################################################
# use of Choi representation
################################################

def _solve_ptm_from_choi_scipy(
        error_mbmat_list, 
        atol = 1e-16, 
        btol = 1e-16, 
        constraint = "pauli", 
        method = "highs", 
        return_res = False,
        presolve = 1,
        tcounts = None,
        tcount_coeff = 0.0,
        dnorm_list = None,
        dnorm_coeff = 0.0
        ):
    n_qubit = int(np.log2(error_mbmat_list[0].shape[0])/2)
    assert constraint in ["pauli", "depol", "XY", "X"], f"{constraint=} not implemented yet."
    
    # 
    A = error_mbmat_list.T.real
    m, n = A.shape

    # 目的関数の係数（1 - Σai[0] のために、最初の行の要素に -1 を掛ける）
    c = -A[0, :]
    if tcounts is not None:
        c += np.array(tcounts) * tcount_coeff
    if dnorm_list is not None:
        c += - np.array(dnorm_list) * dnorm_coeff

    # 制約条件の定義
    # Σpj * ai[k] == 0 に対応する制約を作成
    indices = [4*i + j for i in range(4) for j in range(4) if i!=j] 

    # 不等式制約の追加
    # Σpj * ai[k] >= 0 と Σpj * ai[k] <= 0    
    A_ub = np.vstack([A[indices, :], -A[indices, :]])
    b_ub = np.array([+atol] * len(indices) + [+atol] * len(indices))    
    
    if constraint == "pauli":
        #A_ub = np.vstack([A[indices, :], -A[indices, :]])
        #b_ub = np.array([+atol] * len(indices) + [+atol] * len(indices))
        pass
    elif constraint == "depol":
        # 対角要素に関する制約 (as inequality)
        A_ub2 = np.array([
            (A[1*4 + 1] - A[2*4 + 2]).tolist(),
            (A[2*4 + 2] - A[3*4 + 3]).tolist(),
        ])
        A_ub2 = np.vstack((A_ub2, -A_ub2))
        b_ub2 = [btol for _ in range(2*2)]    

        A_ub = np.vstack((A_ub, A_ub2))
        b_ub = np.concatenate((b_ub, b_ub2))      
        
    elif constraint == "XY":
        # 対角要素に関する制約 (as inequality)
        A_ub2 = np.array([
            (A[1*4 + 1]).tolist(), # confirmed via
        ])
        A_ub2 = np.vstack((A_ub2, -A_ub2))
        b_ub2 = [btol for _ in range(2)]    

        A_ub = np.vstack((A_ub, A_ub2))
        b_ub = np.concatenate((b_ub, b_ub2))              
        
    elif constraint == "X":
        # 対角要素に関する制約 (as inequality)
        A_ub2 = np.array([
            (A[1*4 + 1]).tolist(),
            (A[3*4 + 3]).tolist(),
        ])
        A_ub2 = np.vstack((A_ub2, -A_ub2))
        b_ub2 = [btol for _ in range(2*2)]    

        A_ub = np.vstack((A_ub, A_ub2))
        b_ub = np.concatenate((b_ub, b_ub2))                      
        

    # 確率の条件を満たすための制約
    # Σpj = 1
    A_eq = np.array(np.ones((1, n)))
    b_eq = np.array([1], dtype = float)

    # 確率変数の下限（0以上でなければならない）
    #bounds = [(0, None) for _ in range(n)]

    # 1e-9 seems to be unchangable tolerance value in scipy
    if atol < 1e-9:
        mult = 1e-9/atol
        A_ub *= mult; b_ub *= mult; A_eq *= mult; b_eq *= mult; 

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method = method, options = {"maxiter":100, "presolve":presolve==1})

    try:
        x=res.x
        if return_res:
            return x/sum(x), res
        return x/sum(x)
    except:
        if return_res:
            return None, res
        return None

    #return     