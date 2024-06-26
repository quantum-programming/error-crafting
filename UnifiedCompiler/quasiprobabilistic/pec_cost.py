from scipy.optimize import linprog
import numpy as np
import warnings

PEC_SOLVERS = ["scipy", "cvx", "analytic", "cvxpy"]

def compute_pec_cost(A_mat, b_vec, atol = 1e-6, solver = "analytic", diag_tol = 1e-10, return_solution = False):
    """
    solver = scipy, cvx, cvxpy, analytic
    """
    
    assert solver in PEC_SOLVERS, f"solver must be chosen from {PEC_SOLVERS}"
    if solver == "scipy":
        if return_solution:
            raise NotImplementedError()
        return _compute_gamma_square_scipy(A_mat, b_vec, return_solution = return_solution)
    elif solver in ["cvx", "cvxpy"]:
        if return_solution:
            raise NotImplementedError()
        return _compute_gamma_square_cvx(A_mat, b_vec, atol, return_solution = return_solution)
    elif solver == "analytic":
        #n_qubit = int(np.log2(b_vec.size)/4)
        #target_ptm = b_vec.reshape(4**(n_qubit), 4**n_qubit)
        #assert np.allclose(0, target_ptm - np.diag(target_ptm.diagonal()), atol = diag_tol), "error channel is not Pauli."
        x = np.linalg.pinv(A_mat) @ b_vec
        cost = np.abs(x).sum()**2
        if return_solution:
            return cost, x
        return cost
    elif "diag-analytic":
        ptm = b_vec.reshape(4,4)
        tmps = (1 - ptm.real.diagonal())[1:]
        error_rates = np.linalg.pinv([[0, 2, 2], [2,0,2], [2, 2, 0]]) @ tmps
        return 1 + sum(error_rates)*2


def compute_gamma_factor(A_mat, b_vec, atol = 1e-6, solver = "analytic", diag_tol = 1e-10, return_solution = False):
    """
    solver = scipy, cvx, cvxpy, analytic
    """
    warnings.warn("This function computes gamma^2 instead of gamma. Use 'compute_pec_cost' instead.")
    assert solver in PEC_SOLVERS, f"solver must be chosen from {PEC_SOLVERS}"
    if solver == "scipy":
        if return_solution:
            raise NotImplementedError()
        return _compute_gamma_square_scipy(A_mat, b_vec, return_solution = return_solution)
    elif solver in ["cvx", "cvxpy"]:
        if return_solution:
            raise NotImplementedError()
        return _compute_gamma_square_cvx(A_mat, b_vec, atol, return_solution = return_solution)
    elif solver == "analytic":
        #n_qubit = int(np.log2(b_vec.size)/4)
        #target_ptm = b_vec.reshape(4**(n_qubit), 4**n_qubit)
        #assert np.allclose(0, target_ptm - np.diag(target_ptm.diagonal()), atol = diag_tol), "error channel is not Pauli."
        x = np.linalg.pinv(A_mat) @ b_vec
        cost = np.abs(x).sum()**2
        if return_solution:
            return cost, x
        return cost


def _compute_gamma_square_cvx(A_mat, b_vec, atol = 1e-6, return_solution = False):
    import cvxpy as cvx
    x = cvx.Variable(shape=(A_mat.shape[1]))
    objective = cvx.Minimize(cvx.norm(x, 1))
    constraints = [cvx.norm(A_mat @ x - b_vec, 1) <= atol]
    prob = cvx.Problem(objective, constraints)
    prob.solve()    
    if return_solution:
        return prob.value**2, x
    return prob.value**2

def _compute_gamma_square_scipy(A, b, return_solution = False):
    m, n = A.shape

    A = A.copy().real
    b = b.copy().real
    # c は目的関数の係数ベクトルで、x の L1 ノルムを最小化します
    c = np.concatenate([np.zeros(n), np.ones(n)], axis=0)

    # A_eq と b_eq は、Ax = b および \sum x = 1 の制約を表します
    A_eq = np.hstack([A, np.zeros((m, n))])
    A_eq = np.vstack([A_eq, np.hstack([np.ones((1, n)), np.zeros((1, n))])])
    b_eq = np.hstack([b, [1]])

    # x_i の上限と下限を設定し、y_i = |x_i| を追加します
    bounds = [(None, None) for _ in range(n)] + [(0, None) for _ in range(n)]

    # x - y <= 0 および -x - y <= 0 の制約を追加します
    A_ub = np.vstack([np.hstack([np.eye(n), -np.eye(n)]), np.hstack([-np.eye(n), -np.eye(n)])])
    b_ub = np.zeros(2*n)

    # 最適化を行います
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # 結果を表示します
    if res.success:
        x_solution = res.x[:n]
        #print("Optimal solution found:", x_solution)
    else:
        print("No solution found")

    # x_solution は求める解です
    if return_solution:
        return np.abs(x_solution).sum()**2, x_solution
    return np.abs(x_solution).sum()**2