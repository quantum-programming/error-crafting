from UnifiedCompiler.coherent.synthesis_general_1Q import exact_u_gate

def _run_direct_search(angles, eps):
    import sys
    sys.path.append("./SU2_Clifford_T/")
    import SU2_Compiler
    
    U=exact_u_gate(*angles, as_matrix = True, backend = "qulacs")
    return SU2_Compiler.SU2_Compiler(U, eps)[::-1]