import numpy as np
import copy
from typing import Union, List
import warnings
#from qiskit.quantum_info import  Operator
from qulacs import QuantumCircuit as QC

from UnifiedCompiler.coherent.synthesis_general_1Q import (
    get_u3_angles, 
    exact_u_gate, 
    synthesize_u_gates_in_parallel,
    synthesize_u_gates_in_parallel_direct, 
    generate_epsilon_net_unitaries
    )
from UnifiedCompiler.coherent.synthesis_pauli_rotation import (
    synthesize_single_pauli_rotation_in_parallel, 
    synthesize_single_pauli_rotation_conj_in_parallel, 
    single_qubit_gate
    )

from UnifiedCompiler.coherent.conjugation import conjugate_pauli_rotation, four_fold_conjugation

from UnifiedCompiler.utils.circuit_utils import circuit_to_unitary
from UnifiedCompiler.utils.channel_utils import unitary_to_ptm, operator_norm, diamond_norm_precise
from UnifiedCompiler.probabilistic import solve_ptm

from UnifiedCompiler.coherent.synthesis_pauli_rotation import synthesize_single_qubit_gate
from UnifiedCompiler.coherent.solovay_kitaev import count_t_gates

from UnifiedCompiler.quasiprobabilistic.pec_cost import compute_gamma_factor
from UnifiedCompiler.quasiprobabilistic import A_mat_pauli, A_mat_univ

from UnifiedCompiler.error import PTMCompilationError
#from UnifiedCompiler.coherent.

PAULIROTATIONS = ["Rx", "Ry", "Rz"]

def modify_phase(u_target, u, atol = 1e-2):
    el_target_real = np.copy(u_target)[0, 0].real
    el = np.copy(u)[0, 0]
    if np.isclose(el_target_real, np.real(el), atol = atol):
        return u.copy()
    elif np.isclose(el_target_real, np.real(el * (-1)), atol = atol):
        return u.copy() * (-1)
    elif np.isclose(el_target_real,np.real(el * (-1j)), atol = atol):
        return u.copy() * (-1j)
    elif np.isclose(el_target_real,np.real(el * (1j)), atol = atol):
        return u.copy() * (1j)    
    else:
        return u.copy()

class Unified1QGateCompiler(object):
    def __init__(
        self,
        angle: Union[float, List],
        gate_type,
        error_type,
        solver_type,
        eps,
        ratio_list = None,
        verbose = 1,
        c = None,
        J=1,
        atol_min = 1e-16,
        btol_min = 1e-16,
        pauli_conjugation_number = 2,
        use_magic_basis_only = True,
        n_shift_unitary = None
    ):
        self.angle = angle
        self.gate_type = gate_type
        self.error_type = error_type
        self.solver_type = solver_type
                    
        self.eps = eps
        self.ratio_list = ratio_list
        self.verbose = verbose
        self.c = c
        self.J=J
        if ratio_list is None:
            self.ratio_list = np.linspace(-c, c, J)
        else:
            self.ratio_list = ratio_list

        self.pauli_conjugation_number = pauli_conjugation_number

        # Probabilistic Solver setting
        self.atol_min = atol_min
        self.btol_min = btol_min

        self.use_magic_basis_only = use_magic_basis_only # fast dnorm calculation
        self.n_shift_unitary = n_shift_unitary

        self.initialize()

    def initialize(self):
        if self.gate_type in PAULIROTATIONS:
            assert self.ratio_list is not None
            
            qc_exact = single_qubit_gate(self.angle, self.gate_type, backend="qulacs")
            u_exact = circuit_to_unitary(qc_exact)
            ptm_exact = unitary_to_ptm(u_exact)
            
            self.u_exact = u_exact.copy()
            self.ptm_exact = ptm_exact.copy()
            
            if self.verbose:
                print("====================")
                print("General 1Q gate compiler options: ")
                print(f"gate_type   = {self.gate_type}")
                print(f"error_type  = {self.error_type}")
                print(f"solver_type = {self.solver_type}")
                print(f"eps         = {self.eps}")
                print(f"angle       = {self.angle}")
                print("====================")                     

            
        elif self.gate_type in ["haar", "general", "Rz-as-haar"]:
            assert self.c is not None, "you need c."
            if isinstance(self.angle, np.ndarray):
                self.angle = self.angle.tolist()
            # Get the angles
            #angles = get_u3_angles(u)
            qc_exact = exact_u_gate(*self.angle, backend="qulacs")
            u_exact = circuit_to_unitary(qc_exact)
            ptm_exact = unitary_to_ptm(u_exact)
            
            self.u_exact = u_exact.copy()
            self.ptm_exact = ptm_exact.copy()

            if self.verbose:
                print("====================")
                print("General 1Q gate compiler options: ")
                print(f"gate_type   = {self.gate_type}")
                print(f"error_type  = {self.error_type}")
                print(f"solver_type = {self.solver_type}")
                print(f"eps         = {self.eps}")
                print(f"angle       = {self.angle}")
                print(f"c           = {self.c}")
                print("====================")                    

        self.u_ideal_net_list = None
        self.u_compiled_net_list = None

        self.error_unitary_list = None
        self.error_ptm_list = None
        self.error_ptm = None
        self.Amat = None
        self.rank = None
        
        self.sol_cvx = None


    ##############################################################            
    # Main function
    ##############################################################            
    def run(self):
        self.run_coherent_compilation()
        self.run_probabilistic_compilation()
        self.compute_prob_compilation_error(use_magic_basis_only = self.use_magic_basis_only)

        self.run_quasiprobabilistic_compilation()            
            
    ##############################################################            
    # Coherent synthesis
    ##############################################################    
    #                     
    def run_coherent_compilation(self):
        
        if self.gate_type in PAULIROTATIONS:
            # Deterministic compile
            self._run_deterministic_compilation_pauli()

            # Shifted unitary compilation
            self._run_coherent_compilation_pauli()


        elif self.gate_type in ["general", "haar", "Rz-as-haar"]:
            # Deterministic compile
            self._run_deterministic_compilation_general()
            
            # Shifted unitary compilation
            self._run_coherent_compilation_general()
        
    def _run_deterministic_compilation_general(self):
        angles_exact = get_u3_angles(self.u_exact)
        qc_det = synthesize_u_gates_in_parallel([angles_exact], eps = self.eps/2, backend = "qulacs")[0]
        nt_det = count_t_gates(qc_det)
        u_det = modify_phase(self.u_exact, circuit_to_unitary(qc_det))
        dnorm_det = diamond_norm_precise(self.u_exact, u_det, is_unitary = True)
        ptm_det = unitary_to_ptm(u_det)
        error_ptm_det = unitary_to_ptm(self.u_exact @ (u_det).conj().T)
        
        self.angles_exact = angles_exact
        self.u_det = u_det.copy()
        self.dnorm_det = dnorm_det
        self.tcount_det = nt_det
        self.ptm_det = ptm_det.copy()
        self.error_ptm_det = error_ptm_det.copy()

    def _run_deterministic_compilation_general_direct(self):
        angles_exact = get_u3_angles(self.u_exact)
        #qc_det = synthesize_u_gates_in_parallel([angles_exact], eps = self.eps/2, backend = "qulacs")[0]
        qc_det = synthesize_u_gates_in_parallel_direct([angles_exact], eps = self.eps/2, backend = "qulacs")[0]
        nt_det = count_t_gates(qc_det)
        u_det = modify_phase(self.u_exact, circuit_to_unitary(qc_det))
        dnorm_det = diamond_norm_precise(self.u_exact, u_det, is_unitary = True)
        ptm_det = unitary_to_ptm(u_det)
        error_ptm_det = unitary_to_ptm(self.u_exact @ (u_det).conj().T)
        
        self.angles_exact = angles_exact
        self.u_det = u_det.copy()
        self.dnorm_det = dnorm_det
        self.tcount_det = nt_det
        self.ptm_det = ptm_det.copy()
        self.error_ptm_det = error_ptm_det.copy()        

    def _run_deterministic_compilation_pauli(self):
        # Deterministic compile
        qc_det = synthesize_single_qubit_gate(self.angle, self.gate_type, eps=self.eps/2)
        tcount_det = count_t_gates(qc_det)
        u_det = circuit_to_unitary(qc_det)
        ptm_det = unitary_to_ptm(u_det)
        dnorm_det = diamond_norm_precise(self.u_exact, u_det, is_unitary = True)
        error_ptm_det = self.ptm_exact @ np.linalg.pinv(ptm_det)
        
        self.u_det = u_det.copy()
        self.dnorm_det = dnorm_det
        self.tcount_det = tcount_det
        self.ptm_det = ptm_det.copy()
        self.error_ptm_det = error_ptm_det.copy()        

    def _run_coherent_compilation_general(self):
        u_exact = self.u_exact
        eps = self.eps
        c = self.c
        J=self.J
        c_all_list = np.linspace(0, c, J+1 )[1:]
        error_type = self.error_type
        
        if self.verbose:
            print("\n\n===========================================")
            print("Running coherent compilation for general 1q gate...")
            print("===========================================\n")

        qc_net_list_tot = []
        u_net_list_tot = []
        angles_list = []
        for _c in c_all_list:
            # generate epsilon net by Akibue-san's method
            u_ideal_net_list_tmp = generate_epsilon_net_unitaries(u_exact, eps, _c, error_type = error_type, n_shift_unitary = self.n_shift_unitary)
            angles_list = angles_list + [get_u3_angles(_u) for _u in u_ideal_net_list_tmp]        
            #angles_list 

        self._angles_list = angles_list

        # Compile epsilon net
        qc_compiled_net_list = synthesize_u_gates_in_parallel(angles_list, eps = eps/2, backend = "qulacs", )
        u_compiled_net_list = [circuit_to_unitary(_qc) for _qc in qc_compiled_net_list]
        u_compiled_net_list = [
            _u if operator_norm(_u, u_exact) < operator_norm(_u * (-1), u_exact) 
            else _u * (-1) for _u in u_compiled_net_list 
            ]
            
        qc_net_list_tot = qc_net_list_tot + qc_compiled_net_list
        u_net_list_tot = u_net_list_tot + u_compiled_net_list

        self.u_compiled_net_list = copy.deepcopy(u_net_list_tot)

        #t_counts = np.array([count_t_gates(_qc) for _qc in (qc_compiled_net_list)])        
        t_counts = np.array([count_t_gates(_qc) for _qc in (qc_net_list_tot)])        
        self.t_counts = copy.deepcopy(t_counts)
        
        if self.verbose:
            print("\n\n...done.")
            #print("===========================================")

    def _run_coherent_compilation_general_direct(self):
        u_exact = self.u_exact
        eps = self.eps
        c = self.c
        J=self.J
        c_all_list = np.linspace(0, c, J+1 )[1:]
        error_type = self.error_type
        
        if self.verbose:
            print("\n\n===========================================")
            print("Running coherent compilation for general 1q gate...")
            print("===========================================\n")

        qc_net_list_tot = []
        u_net_list_tot = []
        angles_list = []
        for _c in c_all_list:
            # generate epsilon net by Akibue-san's method
            u_ideal_net_list_tmp = generate_epsilon_net_unitaries(u_exact, eps, _c, error_type = error_type, n_shift_unitary = self.n_shift_unitary)
            angles_list = angles_list + [get_u3_angles(_u) for _u in u_ideal_net_list_tmp]        
            #angles_list 

        self._angles_list = angles_list

        # Compile epsilon net
        qc_compiled_net_list = synthesize_u_gates_in_parallel_direct(angles_list, eps = eps/2, backend = "qulacs", )
        u_compiled_net_list = [circuit_to_unitary(_qc) for _qc in qc_compiled_net_list]
        u_compiled_net_list = [
            _u if operator_norm(_u, u_exact) < operator_norm(_u * (-1), u_exact) 
            else _u * (-1) for _u in u_compiled_net_list 
            ]
            
        qc_net_list_tot = qc_net_list_tot + qc_compiled_net_list
        u_net_list_tot = u_net_list_tot + u_compiled_net_list

        self.u_compiled_net_list_direct = copy.deepcopy(u_net_list_tot)

        #t_counts = np.array([count_t_gates(_qc) for _qc in (qc_compiled_net_list)])        
        t_counts = np.array([count_t_gates(_qc) for _qc in (qc_net_list_tot)])        
        self.t_counts_direct = copy.deepcopy(t_counts)
        
        if self.verbose:
            print("\n\n...done.")
            #print("===========================================")            

        

    def _run_coherent_compilation_pauli(self):
        ratio_list = self.ratio_list
        eps = self.eps
        if self.n_shift_unitary is not None:
            warnings.warn("n_shift_unitary is not valid for Pauli rotation synthesis with Ross-Selinger. We neglect the input.")
        
        if self.verbose:
            print("\n\n===========================================")
            print("Running coherent compilation for 1q Pauli rotation...")
            print("===========================================\n")
        
        angles_list = [self.angle + _ratio * eps  for _ratio in ratio_list]
        gate_type = self.gate_type
        eps = self.eps
        #u_exact = self.u_exact

        qc_compiled_net_list = synthesize_single_pauli_rotation_in_parallel(angles_list, gate_type, eps=eps/2, backend="qulacs")
        u_compiled_net_list = [modify_phase(self.u_exact, circuit_to_unitary(_qc)) for _qc in qc_compiled_net_list]

        # Generate "conjugated" circuit
        if self.pauli_conjugation_number==2:
            qc_compiled_net_list_conj = [conjugate_pauli_rotation(_qc, gate_type = gate_type) for _qc in qc_compiled_net_list]
            qc_compiled_net_list_full = qc_compiled_net_list + qc_compiled_net_list_conj

            u_compiled_net_list_conj = [modify_phase(self.u_exact, circuit_to_unitary(_qc)) for _qc in qc_compiled_net_list_conj]
            u_compiled_net_list_full = u_compiled_net_list + u_compiled_net_list_conj            
        elif self.pauli_conjugation_number == 4:
            qc_compiled_net_list_full = sum([four_fold_conjugation(_qc, gate_type) for _qc in qc_compiled_net_list], [])
            u_compiled_net_list_full = [modify_phase(self.u_exact, circuit_to_unitary(_qc)) for _qc in qc_compiled_net_list_full]

        self.u_compiled_net_list = copy.deepcopy(u_compiled_net_list_full)
        self.u_compiled_net_list_half = copy.deepcopy(u_compiled_net_list)

        t_counts = np.array([count_t_gates(_qc) for _qc in qc_compiled_net_list])        
        self.t_counts_opt_half = copy.deepcopy(t_counts)
        
        t_counts_tot = np.array([count_t_gates(_qc) for _qc in (qc_compiled_net_list_full)])        
        self.t_counts = copy.deepcopy(t_counts_tot)
        
        if self.verbose:
            print("\n\n...done.")
            #print("===========================================")


    ##############################################################            
    #
    # Probabilistic synthesis
    #
    ##############################################################                        

    def run_probabilistic_compilation(self, solver_type = None, atol_min=None, btol_min=None):
        
        if self.gate_type in PAULIROTATIONS:
            self._run_probabilistic_compilation_pauli(solver_type = solver_type, atol_min=atol_min, btol_min=btol_min)
        elif self.gate_type in ["haar", "general", "Rz-as-haar"]:
            self._run_probabilistic_compilation_general(solver_type = solver_type, atol_min=atol_min, btol_min = btol_min)
        else:
            raise NotImplementedError()
            
    def _run_probabilistic_compilation_general(self, solver_type = None, atol_min=None, btol_min = None):
        u_compiled_net_list = self.u_compiled_net_list
        u_exact = self.u_exact
        if solver_type is None:
            solver_type = self.solver_type
        if atol_min is None:
            atol_min = self.atol_min
        if btol_min is None:
            btol_min = self.btol_min
        error_type = self.error_type
        if self.verbose:
            print("\n\n===========================================")
            print("Running probabilistic compilation:")        
            print("===========================================\n")

        error_unitary_list = [_u @ u_exact.conj().T for _u in u_compiled_net_list]
        error_ptm_list = [unitary_to_ptm(_eu) for _eu in error_unitary_list]

        self.error_ptm_list = copy.deepcopy(error_ptm_list)

        sol_cvx = solve_ptm(
            error_ptm_list, 
            solver_type = solver_type, 
            constraint = error_type, 
            atol_min = atol_min, 
            btol_min = btol_min, 
            method = "highs",
            verbose = self.verbose
            )
        error_ptm_opt = np.einsum("i,ijk->jk", sol_cvx, error_ptm_list)


        Amat = np.array([_ptm.reshape(_ptm.size) for _ptm in error_ptm_list]).T.real
        rank = np.linalg.matrix_rank(Amat)

        self.error_ptm_opt = copy.deepcopy(error_ptm_opt)
        self.sol_cvx = copy.deepcopy(sol_cvx)
        self.Amat = copy.deepcopy(Amat)
        self.rank = copy.deepcopy(rank)
        
        if self.verbose:
            print("\n\n...done.")
            #print("===========================================")
            
    def _compile_all_unitary(self):
        u_compiled_net_list = self.u_compiled_net_list
        u_exact = self.u_exact

        # compute error ptms
        error_unitary_list = [_u @ u_exact.conj().T for _u in u_compiled_net_list]
        error_ptm_list = [unitary_to_ptm(_eu) for _eu in error_unitary_list]
        self.error_ptm_list = copy.deepcopy(error_ptm_list)
        self.error_unitary_list = copy.deepcopy(error_unitary_list)

        try:
            error_unitary_list_half = [_u @ u_exact.conj().T for _u in self.u_compiled_net_list_half]
            error_ptm_list_half = [unitary_to_ptm(_eu) for _eu in error_unitary_list_half]
            self.error_ptm_list_half = copy.deepcopy(error_ptm_list_half)        
        except:
            pass

    def compile_all_unitary(self):
        self._run_coherent_compilation_general()

        u_compiled_net_list = self.u_compiled_net_list
        u_exact = self.u_exact

        # compute error ptms
        error_unitary_list = [_u @ u_exact.conj().T for _u in u_compiled_net_list]
        error_ptm_list = [unitary_to_ptm(_eu) for _eu in error_unitary_list]
        self.error_ptm_list = copy.deepcopy(error_ptm_list)
        self.error_unitary_list = copy.deepcopy(error_unitary_list)

        try:
            error_unitary_list_half = [_u @ u_exact.conj().T for _u in self.u_compiled_net_list_half]
            error_ptm_list_half = [unitary_to_ptm(_eu) for _eu in error_unitary_list_half]
            self.error_ptm_list_half = copy.deepcopy(error_ptm_list_half)        
        except:
            pass


    def compile_all_unitary_direct(self):
        self._run_coherent_compilation_general_direct()

        u_compiled_net_list_direct = self.u_compiled_net_list_direct
        u_exact = self.u_exact

        # compute error ptms
        error_unitary_list = [_u @ u_exact.conj().T for _u in u_compiled_net_list_direct]
        error_ptm_list = [unitary_to_ptm(_eu) for _eu in error_unitary_list]
        self.error_ptm_list_direct = copy.deepcopy(error_ptm_list)
        self.error_unitary_list_direct = copy.deepcopy(error_unitary_list)

        try:
            error_unitary_list_half = [_u @ u_exact.conj().T for _u in self.u_compiled_net_list_half]
            error_ptm_list_half = [unitary_to_ptm(_eu) for _eu in error_unitary_list_half]
            self.error_ptm_list_half = copy.deepcopy(error_ptm_list_half)        
        except:
            pass        

    def _run_probabilistic_compilation_pauli(self, solver_type = None, atol_min = None, btol_min = None):
        if self.verbose:
            print("\n\n===========================================")
            print("Running probabilistic compilation:")        
            print("===========================================\n")

        if atol_min is None:
            atol_min = self.atol_min
            if self.verbose:
                print(f"{atol_min=}")
        if btol_min is None:
            btol_min = self.btol_min            
            if self.verbose:
                print(f"{btol_min=}")     

        if solver_type is None:
            solver_type = self.solver_type

        error_type = self.error_type
        self._compile_all_unitary()
        error_ptm_list = copy.deepcopy(self.error_ptm_list)
        error_ptm_list_half = copy.deepcopy(self.error_ptm_list_half)

        # solve SDP
        sol_cvx = solve_ptm(
            error_ptm_list, 
            solver_type = solver_type, 
            constraint = error_type,
            atol_min = atol_min, 
            btol_min = btol_min, 
            method = "highs",
            verbose = self.verbose            
            )
        error_ptm_opt = np.einsum("i,ijk->jk", sol_cvx, error_ptm_list)
        sol_cvx_half = solve_ptm(
            error_ptm_list_half, 
            solver_type = solver_type, 
            constraint = error_type,
            atol_min = atol_min, 
            btol_min = btol_min, 
            method = "highs",
            verbose = self.verbose            
            )
        error_ptm_opt_half = np.einsum("i,ijk->jk", sol_cvx_half, error_ptm_list_half)
        
        # check the rank of SDP
        Amat = np.array([_ptm.reshape(_ptm.size) for _ptm in error_ptm_list]).T.real
        rank = np.linalg.matrix_rank(Amat)
        Amat_half = np.array([_ptm.reshape(_ptm.size) for _ptm in error_ptm_list_half]).T.real
        rank_half = np.linalg.matrix_rank(Amat_half)
        

        # save results
        self.sol_cvx = copy.deepcopy(sol_cvx)
        self.sol_cvx_half = copy.deepcopy(sol_cvx_half)
        self.error_ptm_opt = copy.deepcopy(error_ptm_opt)
        self.error_ptm_opt_half = copy.deepcopy(error_ptm_opt_half)
        
        self.Amat = copy.deepcopy(Amat)
        self.rank = copy.deepcopy(rank)
        self.Amat_half = copy.deepcopy(Amat_half)
        self.rank_half = copy.deepcopy(rank_half)
                
        if self.verbose:
            print("\n\n...done.")
            #print("===========================================")


    ##############################################################            
    #
    # Error analysis
    #
    ##############################################################                                    

    def compute_prob_compilation_error(self, use_magic_basis_only = True):
        if self.gate_type in PAULIROTATIONS:
            self._compute_prob_compilation_error_pauli(use_magic_basis_only = use_magic_basis_only)
        elif self.gate_type in ["haar", "general", "Rz_as_haar"]:
            self._compute_prob_compilation_error_general(use_magic_basis_only = use_magic_basis_only)
            
    def _compute_prob_compilation_error_pauli(self,use_magic_basis_only = True):
        if self.verbose:
            print("\n\n===========================================")
            print("Computing compilation error...")
            print("===========================================\n")

        u_exact = self.u_exact.copy()
        u_det = self.u_det.copy()

        error_ptm_list_full = self.error_ptm_list.copy()
        error_ptm_list_half = self.error_ptm_list_half.copy()
        
        sol_cvx_full = self.sol_cvx.copy()
        sol_cvx_half = self.sol_cvx_half.copy()
        
        error_ptm_opt_full = self.error_ptm_opt.copy()
        error_ptm_opt_half = self.error_ptm_opt_half.copy()

        # Deterministic compilation cost
        error_calculator_full = PTMCompilationError(error_ptm_list_full, sol_cvx_full)
        error_calculator_half = PTMCompilationError(error_ptm_list_half, sol_cvx_half)

        # Probabilistic compilation cost
        dnorm_det = diamond_norm_precise(u_exact, u_det, is_unitary = True)
        if not use_magic_basis_only:
            dnorm_opt_half = diamond_norm_precise(error_ptm_opt_half, unitary_to_ptm(np.diag([1, 1])), scale = 1e9)
            dnorm_opt_full = diamond_norm_precise(error_ptm_opt_full, unitary_to_ptm(np.diag([1, 1])), scale = 1e9)
        dnorm_opt_tot_mb = error_calculator_full.compute_dnorm_mb()            
        
        l1_nondiag_half = np.abs(error_ptm_opt_half - np.diag(error_ptm_opt_half.diagonal())).sum()        
        l1_nondiag_full = np.abs(error_ptm_opt_full - np.diag(error_ptm_opt_full.diagonal())).sum()        
        
        tcount_opt_half = sol_cvx_half @ self.t_counts_opt_half
        tcount_opt_full = sol_cvx_full @ self.t_counts

        self.dnorm_det = dnorm_det
        if not use_magic_basis_only:
            self.dnorm_opt_half = dnorm_opt_half
            self.dnorm_opt_tot = dnorm_opt_full
        self.dnorm_opt_mb = dnorm_opt_tot_mb

        self.l1_nondiag_half = l1_nondiag_half
        self.l1_nondiag_tot = l1_nondiag_full
        
        self.tcount_opt_half = tcount_opt_half
        self.tcount_opt_tot = tcount_opt_full
        
        self.error_calculator_half = copy.deepcopy(error_calculator_half)
        self.error_calculator_tot = copy.deepcopy(error_calculator_full)
        self.error_calculator = copy.deepcopy(error_calculator_full)

        if self.verbose:
            print(f"dnorm_det                 = {dnorm_det:.5e}")
            if not use_magic_basis_only:
                print(f"dnorm_opt_half            = {dnorm_opt_half:.5e}")
                print(f"dnorm_opt_tot             = {dnorm_opt_full:.5e}")
            print(f"dnorm_opt_tot_mb       = {dnorm_opt_tot_mb:.5e}")
            print(f"2*(eps)**2                  = {(self.eps)**2:.5e}")

            print(f"\ntcount_det                 = {self.tcount_det}")
            print(f"tcount_opt_half           = {tcount_opt_half}")
            print(f"tcount_opt_tot            = {tcount_opt_full}")

            print(f"\nl1_nondiag_half         = {l1_nondiag_half:.5e}")
            print(f"l1_nondiag_tot            = {l1_nondiag_full:.5e}")       
            if self.error_type == "depol":
                print(f"diag_error                  = {self.error_calculator.compute_diag_error():.5e}")                            

            print("\n\n..done.")        
            #print("===========================================")
            
    def _compute_prob_compilation_error_general(self, use_magic_basis_only = True):
        if self.verbose:
            print("\n\n===========================================")
            print("Computing compilation error...")
            print("===========================================\n")

        u_exact = self.u_exact
        u_det = self.u_det        

        error_ptm_list_tot = self.error_ptm_list
        sol_cvx_tot = self.sol_cvx
        error_ptm_opt_tot = self.error_ptm_opt

        tcount_opt_tot = sol_cvx_tot @ self.t_counts
        self.tcount_opt_tot = tcount_opt_tot

        # Deterministic compilation cost
        error_calculator_tot = PTMCompilationError(error_ptm_list_tot, sol_cvx_tot)

        # Probabilistic compilation cost
        dnorm_det = diamond_norm_precise(u_exact, u_det, is_unitary = True)
        if not use_magic_basis_only:
            dnorm_opt = diamond_norm_precise(error_ptm_opt_tot, unitary_to_ptm(np.diag([1, 1])))
            self.dnorm_opt = dnorm_opt
        dnorm_opt_mb = error_calculator_tot.compute_dnorm_mb()            
        l1_nondiag_tot = np.abs(error_ptm_opt_tot - np.diag(error_ptm_opt_tot.diagonal())).sum()                

        self.dnorm_det = dnorm_det
        self.dnorm_opt_mb = dnorm_opt_mb

        self.l1_nondiag_tot = l1_nondiag_tot
        self.error_calculator = copy.deepcopy(error_calculator_tot)

        if self.verbose:
            print(f"dnorm_det         = {dnorm_det:.5e}")
            if not use_magic_basis_only:
                print(f"dnorm_opt         = {dnorm_opt:.5e}")
            print(f"dnorm_opt_mb = {dnorm_opt_mb:.5e}")
            print(f"2*(eps)**2 = {2*(self.eps)**2:.5e}")

            print(f"\ntcount_det = {self.tcount_det}")
            print(f"tcount_opt_tot = {tcount_opt_tot}")

            print(f"\nl1_nondiag_tot = {l1_nondiag_tot:.5e}")            

            print("\n\n...done.")
            #print("===========================================")            
        

    ##############################################################            
    #
    # Quasiprobability compile (PEC)
    #
    ##############################################################                                

    def run_quasiprobabilistic_compilation(self, atol = 1e-8, pec_solver = "analytic"):
        if self.verbose:
            print("\n\n===========================================")
            print("Running Quasiprobabilistic compilation...")
            print("===========================================\n")

        target_ptm_opt = np.linalg.pinv(self.error_ptm_opt)
        target_ptm_det = np.linalg.pinv(self.error_ptm_det)

        b_vec_opt = target_ptm_opt.reshape(target_ptm_opt.size)
        b_vec_det = target_ptm_det.reshape(target_ptm_det.size)
        
        if self.error_type == "pauli":            
            A_mat = A_mat_pauli
        elif self.error_type == "depol":            
            A_mat = A_mat_pauli
        else:            
            A_mat = A_mat_univ
            
        cost_pec_opt, x_opt = compute_gamma_factor(A_mat, b_vec_opt, atol = atol, solver = pec_solver, return_solution =True)            
        cost_pec_det, x_det = compute_gamma_factor(A_mat_univ, b_vec_det, atol = atol, solver = pec_solver, return_solution =True)

        self.cost_pec_opt = cost_pec_opt
        self.cost_pec_det = cost_pec_det

        self.x_opt = x_opt.copy()
        self.x_det = x_det.copy()

        self.A_mat_opt = A_mat.copy()
        self.A_mat_det = A_mat_univ.copy()
        
        if self.verbose:
            print("Mitigating Solovay-Kitaev error:")
            print(f"\nDeterministic + Naive PEC cost : {cost_pec_det:.8f}")
            #print()
            print(f"Optimized      + Pauli PEC cost : {cost_pec_opt:.8f}")
            #print(f"Optimized      + Naive PEC cost : {cost_opt_univ:.8f}")            
            print("\n\n...done.")
            #print("===========================================")