from fire import Fire
import numpy as np
from UnifiedCompiler.utils.channel_utils import (
    diamond_norm,
    operator_norm
)

from UnifiedCompiler.coherent import (
    generate_epsilon_net_circuits, 
    generate_epsilon_net_unitaries,
    count_t_gates,
)

from UnifiedCompiler.utils import circuit_to_unitary

from UnifiedCompiler.coherent.synthesis_general_1Q import get_u3_angles, exact_u_gate
import tqdm
from UnifiedCompiler import Unified1QGateCompiler

def generate_angles_from_haar_random_1Q(n_data, seed=None):
    from qulacs import QuantumCircuit as QC
    if seed is not None:
        np.random.seed(seed)
    seed_list = np.random.randint(0, 12345678, size = n_data)
    angles_list = []
    for _ in range(n_data):
        # Generate Haar random
        qc = QC(1)
        
        qc.add_random_unitary_gate([0], seed = seed_list[_])
        u = circuit_to_unitary(qc)

        # Get the angles
        angles = get_u3_angles(u)
        #qc_exact = exact_u_gate(*angles, )
        #u_exact = Operator(qc_exact).data        
        angles_list.append(np.copy(angles))
    return np.copy(angles_list)

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

from UnifiedCompiler.probabilistic import solve_ptm, solve_ptm_from_choi

def main(
    gate_type ="haar",
    error_type = "pauli",
    solver_type = "scipy",
    constraint = "X",
    presolve = -1,
    verbose = 0,
    eps_list = [1e-5, 1e-4],
    c_list = [1.5, 2.0, 3.0, 4.0, ],
    J_list = [1,2],
    n_shift_unitary_list = [7,9],
    n_data = 2,
    seed=1234,
    save_each_J=False,
    save_each_c = False,
    save_each_nshift = False,
    nondiag_l1_thres=1e-12,
    atol_min = 1e-15,
    btol_min = 1e-12
):
    
    print(f"{gate_type=}")
    print(f"{error_type=}")
    print(f"{solver_type=}")
    #print(f"{threshold=}")
    print(f"{eps_list=}")

    print(f"{constraint=}")
    print(f"{presolve=}")
    
    #print(f"{J_list=}")
    print(f"{n_data=}")
    print(f"{verbose=}")
    print(f"{seed=}")

    print(f"{c_list=}")
    print(f"{J_list=}")
    print(f"{n_shift_unitary_list=}")
    print(f"{save_each_J=}")    
    print(f"{save_each_c=}")    
    print(f"{save_each_nshift=}")    


    #raise Exception("shall we compute success rate or repetition count?")
    # repetition count is faster,,

    # initialize
    #count_result = {}
    #count_result["eps_list"] = eps_list    
    #count_result["c_list"] = c_list
    #count_result["J_list"] = J_list    
    nondiag_l1_data = {}
    nondiag_l1_data.update(eps_list=eps_list, c_list = c_list, J_list = J_list, n_shift_unitary_list = n_shift_unitary_list)

    diag_error_data = {}
    diag_error_data.update(eps_list=eps_list, c_list = c_list, J_list = J_list, n_shift_unitary_list = n_shift_unitary_list)

    dnorm_data = {} 
    dnorm_data.update(eps_list=eps_list, c_list = c_list, J_list = J_list, n_shift_unitary_list = n_shift_unitary_list)

    success_data = {}
    success_data.update(eps_list=eps_list, c_list = c_list, J_list = J_list, n_shift_unitary_list = n_shift_unitary_list)


    # TODO: import from somewhere?
    PAULIROTATIONS = ["Rx", "Ry", "Rz"]
    if gate_type in PAULIROTATIONS and constraint == "depol":
        raise Exception("Run calc_compile_success_rate_pauli_depol.py instead.")

    if gate_type in ["haar", "general"]:
        angles_list = generate_angles_from_haar_random_1Q(n_data, seed=seed)
        
    elif gate_type in PAULIROTATIONS:
        raise NotImplementedError()
        angles_list = 2 * np.pi * np.random.random(size = n_data)
        param_list = J_list
    else:
        raise Exception(f"{gate_type=} not implemented.")
    
    

    for eps in tqdm.tqdm(eps_list):
        print("\n\n====================================")
        print(f"eps = {eps:.2e}")
        print("====================================")
        #for j, param in enumerate(param_list):
        for J in J_list:
            for c in c_list:
                for n_shift_unitary in n_shift_unitary_list:
                    key = f"eps={eps:.2e}, J={J}, c={c}, n_shift_unitary={n_shift_unitary}"

                    #count_array = [] 
                    
                    nondiag_l1_array = [] 
                    diag_error_array = []  
                    dnorm_array = []
                    success_array = []
                    for i in range(n_data):
                        try:
                            compiler = Unified1QGateCompiler(
                                angles_list[i], 
                                gate_type, 
                                error_type, 
                                solver_type, 
                                eps = eps, 
                                c=c, 
                                verbose = verbose, 
                                J=J,
                                n_shift_unitary=n_shift_unitary
                                )


                            ##compiler.run_coherent_compilation()
                            #compiler.run_probabilistic_compilation()
                            #compiler.run_probabilistic_compilation(atol_min = eps**2, btol_min = min([1e-3, eps**2 * 1000]))
                            #compiler.run_probabilistic_compilation(atol_min = min([1e-14, eps**2]), btol_min = min([1e-7, eps**2 * 1000]))
                            #compiler.compute_prob_compilation_error()
                            #nondiag_l1 = compiler.error_calculator.compute_nondiag_l1()
                            #diag_error = compiler.error_calculator.compute_diag_error()
                            #dnorm = compiler.error_calculator.compute_dnorm_mb()

                            compiler.compile_all_unitary()                        
                            

                            error_ptm_list = compiler.error_ptm_list
                            probs = solve_ptm(error_ptm_list,solver_type = solver_type, constraint = constraint, presolve = presolve, atol_min = atol_min, btol_min =btol_min)
                            error_ptm_opt = np.einsum("i,ijk->jk", probs, error_ptm_list)


                            #error_ptm_opt = compiler.error_ptm_opt
                            error_rates = np.linalg.pinv([[0, 2, 2], [2,0,2], [2, 2, 0]]) @ (1 - error_ptm_opt.real.diagonal())[1:]

                            
                            success = is_success(error_ptm_opt,  eps, constraint, thres_l1 = nondiag_l1_thres)
                            nondiag_l1 = np.sum(np.abs(error_ptm_opt - np.diag(error_ptm_opt.diagonal())))
                            ptm_iden = unitary_to_ptm(np.diag([1, 1]))
                            dnorm = diamond_norm_precise(error_ptm_opt, ptm_iden, scale = 1e4)

                            print(f"{key=}, nondiag_l1={nondiag_l1:.2e}, success={success}")
                            nondiag_l1_array.append(nondiag_l1)
                            #diag_error_array.append(diag_error)
                            dnorm_array.append(dnorm)
                            success_array.append(int(success))
                        except:
                            print(f"{key=},skipped")                        


                    nondiag_l1_data[key] = nondiag_l1_array
                    diag_error_data[key] = diag_error_array    
                    dnorm_data[key] = dnorm_array
                    success_data[key] = success_array

                            
                        #count_array.append(count)

                    #count_result[key] = count_array

                    import json
                    import os

                    if not os.path.exists("results/success_rate"):
                        os.makedirs("./results/", exist_ok=True)
                        os.makedirs("./results/success_rate", exist_ok=True)

                    data = {}    
                    data.update(nondiag_l1_data=nondiag_l1_data, dnorm_data = dnorm_data, success_data = success_data)
                    filename = f"results/success_rate/success_rate_{gate_type}_{constraint}_ndata_{n_data}_cfilled_eps_{eps:.2e}"
                    if save_each_J:
                        filename += f"_J_{J}"            
                    if save_each_c:
                        filename += f"_c_{c}"            
                    if save_each_nshift:
                        filename += f"_nshift_{n_shift_unitary}"            

                    if solver_type != "scipy":
                        filename += f"_{solver_type}"
                    filename += ".json"
                    json.dump(data, open(filename, "w"))
                    print(f"...saved as {filename}")

            


if __name__ == "__main__":
    Fire(main)