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


def main(
    gate_type ="haar",
    error_type = "pauli",
    solver_type = "scipy",
    verbose = 0,
    eps_list = [1e-5, 1e-4],
    c_list = [1.5, 2.0, 3.0, 4.0, ],
    J_list = [1,2],
    n_data = 2,
    seed=1234,
    save_each_eps=True,
    l1_threshold = 1e-14
):
    
    print(f"{gate_type=}")
    print(f"{error_type=}")
    print(f"{solver_type=}")
    #print(f"{threshold=}")
    print(f"{eps_list=}")
    
    #print(f"{J_list=}")
    print(f"{n_data=}")
    print(f"{verbose=}")
    print(f"{seed=}")

    print(f"{c_list=}")
    print(f"{J_list=}")
    print(f"{save_each_eps=}")

    #raise Exception("shall we compute success rate or repetition count?")
    # repetition count is faster,,

    # initialize
    #count_result = {}
    #count_result["eps_list"] = eps_list    
    #count_result["c_list"] = c_list
    #count_result["J_list"] = J_list    
    nondiag_l1_data = {}
    nondiag_l1_data["eps_list"] = eps_list
    nondiag_l1_data["c_list"] = c_list
    nondiag_l1_data["J_list"] = J_list

    dnorm_data = {}
    dnorm_data.update(eps_list = eps_list, c_list = c_list, J_list = J_list)

    dnorm_det_data = {}
    dnorm_det_data.update(eps_list = eps_list, c_list = c_list, J_list = J_list)

    tcount_data = {}
    tcount_data.update(eps_list = eps_list, c_list = c_list, J_list = J_list)


    # TODO: import from somewhere?
    PAULIROTATIONS = ["Rx", "Ry", "Rz"]
    if gate_type in PAULIROTATIONS and error_type == "depol":
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
                key = f"eps={eps:.2e}, J={J}, c={c}"

                #count_array = [] 
                
                nondiag_l1_array = [] 
                #diag_error_array = []
                dnorm_array = []
                dnorm_det_array = []
                tcount_array = []
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
                            J=J
                            )


                        compiler.run_coherent_compilation()
                        #compiler.run_probabilistic_compilation(atol_min = eps**2, btol_min = min([1e-3, eps**2 * 1000]))
                        compiler.run_probabilistic_compilation(atol_min = min([1e-14, eps**2]), btol_min = min([1e-7, eps**2 * 1000]))
                        compiler.compute_prob_compilation_error()
                        nondiag_l1 = compiler.error_calculator.compute_nondiag_l1()
                        dnorm = compiler.error_calculator.compute_dnorm_mb()
                        dnorm_det = compiler.dnorm_det
                        tcount_tot = compiler.tcount_opt_tot
                        #diag_error = compiler.error_calculator.compute_diag_error()

                        print(f"{key=}, nondiag_l1={nondiag_l1:.2e}, dnorm={dnorm:.2e}")
                        nondiag_l1_array.append(nondiag_l1)
                        dnorm_array.append(dnorm)
                        dnorm_det_array.append(dnorm_det)
                        tcount_array.append(tcount_tot)
                    except:
                        print(f"{key=},skipped")                        


                nondiag_l1_data[key] = nondiag_l1_array
                dnorm_data[key] = dnorm_array
                dnorm_det_data[key] = dnorm_det_array
                tcount_data[key] = tcount_array
                        
                    #count_array.append(count)

                #count_result[key] = count_array

            import json
            import os

            if not os.path.exists("results/dnorm_scaling"):
                os.makedirs("./results/", exist_ok=True)
                os.makedirs("./results/dnorm_scaling", exist_ok=True)

            data = {"nondiag_l1_data":nondiag_l1_data, "dnorm_data":dnorm_data, "dnorm_det_data":dnorm_det_data, "tcount_data":tcount_data}

            filename = f"results/dnorm_scaling/dnorm_scaling_{gate_type}_{error_type}_ndata_{n_data}_cfilled"
            if save_each_eps:
                filename += f"_eps_{eps:.2e}"
            if solver_type != "scipy":
                filename += f"_{solver_type}"
            filename += ".json"
            json.dump(data, open(filename, "w"))
            print(f"...saved as {filename}")

            


if __name__ == "__main__":
    Fire(main)