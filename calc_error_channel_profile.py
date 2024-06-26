from fire import Fire

import numpy as np
from UnifiedCompiler.probabilistic import solve_ptm, solve_ptm_from_choi
from UnifiedCompiler.error import diamond_norm_precise, diamond_norm_from_choi
from UnifiedCompiler import Unified1QGateCompiler
import tqdm

def calc_nondiag_l1(error_ptm):
    return np.abs(error_ptm - np.diag(error_ptm.diagonal())).sum()

from UnifiedCompiler.utils import diamond_norm_precise
from UnifiedCompiler.utils import unitary_to_ptm
def is_success(error_ptm_opt, eps, constraint, thres_l1 = 1e-12, z_rate_thres = 0.01, depol_thres = 0.01):
    nondiag_l1 = np.sum(np.abs(error_ptm_opt - np.diag(error_ptm_opt.diagonal())))
    ptm_iden = unitary_to_ptm(np.diag([1, 1]))
    dnorm = diamond_norm_precise(error_ptm_opt, ptm_iden, scale = 1e4)
    err_rate = np.linalg.pinv([[0, 2, 2], [2,0,2], [2, 2, 0]]) @ (1 - error_ptm_opt.real.diagonal())[1:]
    if constraint == "pauli":
        return np.abs(nondiag_l1) <= thres_l1 and 0 <= dnorm <= eps**2 * 10 and min(err_rate) >= 0
    elif constraint == "XY":
        return np.abs(nondiag_l1) <= thres_l1 and 0 <= dnorm <= eps**2 * 10 and min(err_rate) >= 0 and err_rate[2]/sum(err_rate) <= z_rate_thres
    elif constraint == "X":
        return np.abs(nondiag_l1) <= thres_l1 and 0 <= dnorm <= eps**2 * 10 and min(err_rate) >= 0 and err_rate[2]/sum(err_rate) <= z_rate_thres and err_rate[1]/sum(err_rate) <= z_rate_thres
    elif constraint == "depol":
        err_rate_dev = max(err_rate)/sum(err_rate) - min(err_rate)/sum(err_rate)
        return np.abs(nondiag_l1) <= thres_l1 and 0 <= dnorm <= eps**2 * 10 and min(err_rate) >= 0 and err_rate_dev <= depol_thres


def main(
    c=3,
    J=3,
    eps = 1e-4,
    atol_min = 1e-15,
    btol_min = 1e-12,
    method = "highs",
    gate_type = "haar",
    error_type = "depol",
    n_data = 2,
    verbose = 0,
    seed = 12345,
    z_thres = 0.01,
    n_shift_unitary = 9
):
    np.random.seed(seed)

    print(f"{c=}")
    print(f"{J=}")
    print(f"eps={eps:.2e}")
    print(f"atol_min={atol_min:.2e}")
    print(f"btol_min={btol_min:.2e}")
    print(f"{n_data=}")
    print(f"z_thres={z_thres:.2e}")
    print(f"n_shift_unitary={n_shift_unitary:.2e}")


    if gate_type == "haar":
        angles = np.random.random(size = (n_data, 3)) * 2 * np.pi
    else:
        angles = np.random.random(size = n_data) * 2 * np.pi


    result_data = {}
    CONSTRAINTS = ["pauli", "XY", "X", "depol"]
    SOLVER_TYPES = ["scipy-choi", "gurobi-choi"]
    pauli_conjugation_number = 4
    iden_ptm = unitary_to_ptm(np.diag([1, 1]))
    #solver_type = "gurobi-choi"

    for constraint in CONSTRAINTS:
        result_data[constraint] = {}
        result_data[constraint].update(nondiag_l1 = [], error_rates = [], error_rate_ratio = [], dnorm = [], success = [], setup = [])
        
    for angle in tqdm.tqdm(angles):
        compiler = Unified1QGateCompiler(
            #angles_list[i], 
            angle,
            gate_type, 
            error_type, 
            SOLVER_TYPES[0], 
            eps = eps, 
            c=c, 
            J=J,
            verbose = verbose, 
            pauli_conjugation_number = pauli_conjugation_number,
            n_shift_unitary = n_shift_unitary
            )


        compiler.run_coherent_compilation()
        compiler._compile_all_unitary()
        #compiler.run_probabilistic_compilation()

        error_ptm_list = compiler.error_ptm_list
        print()

        for constraint in CONSTRAINTS:
            succ = False
            for solver_type in SOLVER_TYPES:
                probs = solve_ptm(error_ptm_list, constraint = constraint, atol_min = atol_min, solver_type = solver_type, btol_min = btol_min, )
                error_ptm_opt = np.einsum("i,ijk->jk", probs, error_ptm_list)
                error_rates = np.linalg.pinv([[0, 2, 2], [2,0,2], [2, 2, 0]]) @ (1 - error_ptm_opt.real.diagonal())[1:]
                dnorm = diamond_norm_precise(error_ptm_opt, iden_ptm, scale = 1e4)
                #dnorm = sum(error_rates*2)
                nondiag_l1 = calc_nondiag_l1(error_ptm_opt)

                succ = is_success(error_ptm_opt, eps, constraint, z_rate_thres=z_thres)

                if succ:
                    break
                #if succ:
                #    break

            print(f"\n{constraint} solver : {error_rates}")
            print(f"                     : {error_rates/sum(error_rates)}")
            print(f"       dnorm =  {sum(error_rates*2)}")
            print(f"nondiag_l1 =  {nondiag_l1}")        
            #print(f"{presolve=}, {solver_type=}")        
            print(f"success ? {succ}")        

            result_data[constraint]["nondiag_l1"].append(nondiag_l1)
            result_data[constraint]["error_rates"].append(error_rates.tolist())
            result_data[constraint]["error_rate_ratio"].append((error_rates/sum(error_rates)).tolist())
            result_data[constraint]["dnorm"].append(dnorm)
            result_data[constraint]["success"].append(int(succ))
            #result_data[constraint]["setup"].append((presolve, solver_type))

    import json
    import os

    if not os.path.exists("results/error_profile"):
        os.makedirs("./results/", exist_ok=True)
        os.makedirs("./results/error_profile", exist_ok=True)

    #data = {"nondiag_l1_data":nondiag_l1_data, "dnorm_data":dnorm_data, "dnorm_det_data":dnorm_det_data, "tcount_data":tcount_data}

    filename = f"results/error_profile/error_profile_{gate_type}_c_{c}_J_{J}_eps_{eps:.2e}_ndata_{n_data}"
    if z_thres != 1e-2:
        filename += f"_zthres_{z_thres:.2e}"

    filename += ".json"
    json.dump(result_data, open(filename, "w"))
    print(f"...saved as {filename}")    

if __name__ == "__main__":
    Fire(main)