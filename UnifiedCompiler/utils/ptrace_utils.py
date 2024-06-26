import numpy as np

###############################
# channel ptrace
##############################
# see test/test_ptrace.py for usage

def take_ptrace_ptm(ptm, sites):
    """
    partial trace over ancilla assuming their initialization in |0>.
    see test/test_ptrace.py for usage.
    
    args:
        ptm : pauli transfer matrix
        sites : sites to be leaved non-traced out
    """
    assert sites == sorted(sites), "sort sites to avoid bug."
    n_qubit = int(np.log2(ptm.shape[0]))//2
    
    nq_new = len(sites)
    n_traced = n_qubit - nq_new
    ptm_new = np.zeros((4**nq_new, 4**nq_new), dtype = complex)
    for i in range(4**nq_new):
        qstring_i = np.base_repr(i, base = 4).zfill(nq_new)
        dict_i = {site : qstring_i[nq_new - 1 - sites.index(site)] for site in sites}
        
        qlist = ["0" for _ in range(n_qubit)]
        for site in sites:
            qlist[n_qubit - 1 - site] = dict_i[site]
        args_i = [int("".join(qlist), base = 4)]
        #args_i = traced_out_args_as_id(i_site, sites, n_qubit)
        
        for j in range(4**nq_new):
            #args_j = traced_out_args_as_id_and_Z(j_site, sites, n_qubit)    
            qstring_j = np.base_repr(j, base = 4).zfill(nq_new)
            dict_j = {site : qstring_j[nq_new - 1 - sites.index(site)] for site in sites}
            
            args_j = traced_out_args_id_and_Z(dict_j, sites, n_qubit)
            #print(args_i, args_j)
            #print(sum(ptm[args_i, :][:, args_j]))
            #ptm_new[i, j] = np.sum(ptm[args_i, :][:, args_j])/(2**n_traced)
            ptm_new[i, j] = np.sum(ptm[args_i, :][:, args_j])
    return ptm_new

def traced_out_args_id_and_Z(i_dict, sites, n_qubit):
    #raise Exception("there is bug here")
    traced_args = []
    for arg in range(4**n_qubit):
        traced_site_qstrings = [qstr for i, qstr in enumerate(np.base_repr(arg, base = 4).zfill(n_qubit)[::-1]) if i not in sites]
        if "1" in traced_site_qstrings or "2" in traced_site_qstrings:
            continue
        if not is_traced(arg, i_dict, n_qubit):
            continue
            
        traced_args.append(arg)
    return traced_args

def is_traced(arg, arg_site_dict, n_qubit):
    qstring = np.base_repr(arg, base = 4).zfill(n_qubit)
    for site, qarg in arg_site_dict.items():
        if int(qstring[n_qubit - 1 - site]) != int(qarg):
            return False
    return True


###############################
# state ptrace
##############################

def take_ptrace_state(state, sites):
    """
    Take partial trace of quantum state.
    Note that this is slow because we utilize qutip
    """
    if len(state.shape) == 1:
        rho = np.outer(state, state.conj())
    elif len(state.shape) == 2:
        rho = state.copy()

    import qutip as qt        
    n_qubit = rho.shape[0].bit_length() - 1
    rho_qt = qt.Qobj(rho, dims = [[2,]*n_qubit, [2,]*n_qubit])
    rho_qt = rho_qt.permute(range(n_qubit)[::-1])
    return rho_qt.ptrace(sites).full()
