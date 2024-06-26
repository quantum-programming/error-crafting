
import scipy
from UnifiedCompiler.utils.channel_utils import unitary_to_ptm
import numpy as np
import itertools

import copy
from UnifiedCompiler.utils.channel_utils import ptm_to_mb_vec
def solve_ptm_gurobi(
    error_ptm_list,
    constraint ="pauli", 
    atol_min = 1e-17,
    btol_min = 1e-17, 
    verbose = 0, 
    return_res = False, 
    return_tol = False,
    tcounts = None,
    tcount_coeff = 0.01
):
    
#def solve_ptm_pauli_from_choi(error_ptm_list, atol_min = 1e-17, btol_min = 1e-17, verbose = 0, return_res = False, return_tol = False):
    _atol = copy.deepcopy(atol_min)
    _btol = copy.deepcopy(btol_min)
    solution = None

    if atol_min > 0:
        while solution is None:
            solution, res = _solve_ptm_gurobi(error_ptm_list, atol = _atol, btol = _btol, constraint = constraint, return_res = True, tcounts = tcounts, tcount_coeff = tcount_coeff)
            
            if solution is None:
                if verbose > 1:
                    print(f"raising from _atol = {_atol:.2e}")
                _atol *= np.sqrt(10)
                _btol *= np.sqrt(10)
    else:
        solution, res = _solve_ptm_gurobi(
            error_ptm_list, 
            atol = _atol, 
            btol = _btol, 
            constraint = constraint, 
            return_res = True,
            tcounts = tcounts,
            tcount_coeff = tcount_coeff
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

def _solve_ptm_gurobi(
        error_ptm_list, 
        constraint = "pauli", 
        verbose = 0, 
        atol = 1e-16, 
        btol = 1e-16,
        presolve = 0,
        return_res = False,
        tcounts = None,
        tcount_coeff = 0.01
        ):
    import gurobipy as gp
    from gurobipy import GRB        
    from gurobipy import quicksum
    
    n_qubit = int(np.log2(error_ptm_list[0].shape[0])/2)
    A = scipy.sparse.csc_matrix(np.array([_ptm.reshape(_ptm.size) for _ptm in error_ptm_list]).T.real)
    #A = scipy.sparse.hstack([A_tmp_sp, -A_tmp_sp])
    b = unitary_to_ptm(np.identity(2**n_qubit)).reshape(16**n_qubit).real
    
    m,n = A.shape
    
    # モデルの生成
    model = gp.Model("ptm")
    if not verbose:
        model.Params.LogToConsole = 0

    # 変数
    x = model.addMVar(shape = n, lb = 0, obj = 0.0) # x_i^- の生成
    z = model.addMVar(shape=m, lb = 0) # z_i の生成

    # 制約 (1) : |A x - b|_1 < eps 
    model.addConstrs(((A@x - b)[i] <= z[i] for i in range(m)), )
    model.addConstrs(((A@x - b)[i] >= -z[i] for i in range(m)), )
    
    #model.addConstr(quicksum(x) >= 1 - 1e-17)
    model.addConstr(quicksum(x) == 1)    
    
    if constraint in ["pauli", "depol", "XY", "X"]:
        nondiag_args = [(i, j) for (i, j) in itertools.permutations(range(1, 4**n_qubit), 2)]
        model.addConstrs(((A@x)[arg[0] * (4**n_qubit) + arg[1]] >= -atol for arg in nondiag_args), )
        model.addConstrs(((A@x)[arg[0] * (4**n_qubit) + arg[1]] <= atol for arg in nondiag_args), )

    if constraint == "depol":
        model.addConstr((A@x)[1*4+1] - (A@x)[2*4 + 2]<= btol)
        model.addConstr((A@x)[1*4+1] - (A@x)[2*4 + 2]>= -btol)
        model.addConstr((A@x)[2*4+2] - (A@x)[3*4 + 3]<= btol)
        model.addConstr((A@x)[2*4+2] - (A@x)[3*4 + 3]>= -btol)

        
    if constraint in ["XY"]:
        model.addConstr((A@x)[1*4+1] + (A@x)[2*4 + 2] - (A @x)[3*4+3] <= 1 + btol)
        model.addConstr((A@x)[1*4+1] + (A@x)[2*4 + 2] - (A @x)[3*4+3] >= 1 - btol)
        
    if constraint == "X":
        model.addConstr(A[1*4 + 1]@x <= 1 + btol)
        model.addConstr(A[1*4 + 1]@x >= 1 - btol)


    # 目的関数    
    if tcounts is None:
        model.setObjective(quicksum(z[i] for i in range(m)), GRB.MINIMIZE)
    else:
        model.setObjective(quicksum(z[i] for i in range(m)) + z @ tcounts * tcount_coeff, GRB.MINIMZE)

    # モデルの最適化
    if presolve != -1:
        model.Params.Presolve = presolve
    model.optimize()
    
    if return_res:
        return x.X, x
    return x.X

def _solve_ptm_gurobi_from_choi(
    error_mbmat_list, 
    atol = 1e-16, 
    btol = 1e-16, 
    constraint = "pauli", 
    return_res = False,
    verbose = 0,
    presolve = 0,
    tcounts = None,
    tcount_coeff = 0.0,
    dnorm_list = None,
    dnorm_coeff = 0.0
   ):
    
    import gurobipy as gp
    from gurobipy import GRB        
    from gurobipy import quicksum
    
    n_qubit = int(np.log2(error_mbmat_list[0].shape[0])/4)
    A = scipy.sparse.csc_matrix(error_mbmat_list.T.real)
    #A = scipy.sparse.hstack([A_tmp_sp, -A_tmp_sp])
    b = unitary_to_ptm(np.identity(2**n_qubit)).reshape(16**n_qubit).real
    
    m,n = A.shape
    
    # モデルの生成
    model = gp.Model("ptm")
    if not verbose:
        model.Params.LogToConsole = 0    

    # 変数
    x = model.addMVar(shape = n, lb = 0.0, obj = 0.0) # x_i^- の生成
    z = model.addVar( lb = 0.0, obj = 0.0) # x_i^- の生成
    #z = model.addMVar(shape=m, lb = 0) # z_i の生成        

    if constraint in ["pauli", "depol", "XY", "X"]:
        indices = [4*i + j for i in range(4) for j in range(4) if i!=j] 
    elif constraint in ["Xnew"]:
        indices = [4*i + j for i in range(4) for j in range(4) if i!=j and ((i,j) not in [(0,2), (2, 0)])]  
    elif constraint in ["XYnew"]:
        indices = [4*i + j for i in range(4) for j in range(4) if i!=j and ((i,j) not in [(0,2), (2, 0), (0, 3), (3, 0)])]  


    model.addConstr(quicksum(x) == 1)

    # モデルの最適化
    if presolve != -1:
        model.Params.Presolve = presolve

    if constraint in ["pauli", "depol", "XY", "X", "Xnew", "XYnew"]:
        model.addConstrs((A[ind]@x <= atol for ind in indices))
        model.addConstrs((A[ind]@x >= -atol for ind in indices))
        
    if constraint == "depol":
        model.addConstr(A[1*4 + 1]@x - A[2*4+2]@x <= btol)
        model.addConstr(A[1*4 + 1]@x - A[2*4+2]@x >= -btol)
        model.addConstr(A[2*4 + 2]@x - A[3*4+3]@x <= btol)
        model.addConstr(A[2*4 + 2]@x - A[3*4+3]@x >= -btol)
        
        objective = -A[0, :]@x
        #model.setObjective(-A[0, :]@x, GRB.MINIMIZE)


    if constraint in ["XY"]:
        model.addConstr(A[1*4+1]@x <= btol)
        model.addConstr(A[1*4+1]@x >= -btol)
        
        objective = -A[0, :]@x
        #model.setObjective(-A[0, :]@x, GRB.MINIMIZE)


    if constraint == "X":
        model.addConstr(A[1*4+1]@x <= btol)
        model.addConstr(A[1*4+1]@x >= -btol)        
        model.addConstr(A[3*4+3]@x <= btol)
        model.addConstr(A[3*4+3]@x >= -btol)        
        
        objective = -A[0, :]@x
        #model.setObjective(-A[0, :]@x, GRB.MINIMIZE)     

    if  constraint == "Xnew":
        model.addConstr(A[1*4+1]@x <= z)
        model.addConstr(A[1*4+1]@x >= -z)        
        model.addConstr(A[3*4+3]@x <= z)
        model.addConstr(A[3*4+3]@x >= -z)        
        
        objective = z
        #model.setObjective(z, GRB.MINIMIZE)        

    if constraint in ["XYnew"]:
        model.addConstr(A[1*4+1]@x <= z)
        model.addConstr(A[1*4+1]@x >= -z)
        
        objective = z
        #model.setObjective(z, GRB.MINIMIZE)        

    if tcounts is not None:
        objective += np.array(tcounts) @ x * tcount_coeff
    if dnorm_list is not None:
        objective += - np.array(dnorm_list) @ x * dnorm_coeff

    
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    

    try:
        probs=x.X
        if return_res:
            return probs/sum(probs), x
        return probs/sum(probs)
    except:
        if return_res:
            return None, x
        return None