import numpy as np
from UnifiedCompiler.utils.channel_utils import ptm_to_choi_MB
from UnifiedCompiler.utils.channel_utils import diamond_norm_precise, diamond_norm_from_choi

class PTMCompilationError(object):
    def __init__(
        self,
        error_ptm_list,
        coeffs,
    ):
        self.error_ptm_opt = np.sum([coeffs[i] * error_ptm_list[i] for i in range(len(coeffs))], axis = 0)
        self.error_ptm_list = np.copy(error_ptm_list)
        self.coeffs = np.copy(coeffs)

    def compute_dnorm_mb(self):
        
        choi_MB_opt = ptm_to_choi_MB(self.error_ptm_opt)
        dnorm_opt_mb = (2 - choi_MB_opt[0, 0]).real
        return dnorm_opt_mb
        
    def compute_nondiag_l1(self):
        nondiag_L1_opt_ideal = np.abs(self.error_ptm_opt - np.diag(self.error_ptm_opt.diagonal())).sum()
        return nondiag_L1_opt_ideal
    
    def compute_diag_error(self):
        ptm_opt_diag  = self.error_ptm_opt.diagonal()
        diag_diff_opt = np.abs(ptm_opt_diag[1,] - ptm_opt_diag[2,]) 
        + np.abs(ptm_opt_diag[1, ] - ptm_opt_diag[3]) 
        + np.abs(ptm_opt_diag[3,] - ptm_opt_diag[2,])
        return diag_diff_opt
    
    def get_error_ptm(self):
        return self.error_ptm_opt.copy()
    
def diamond_norm_mb(error_ptm):
    choi_MB_opt = ptm_to_choi_MB(error_ptm)
    dnorm_opt_mb = (2 - choi_MB_opt[0, 0]).real
    return dnorm_opt_mb

def compute_nondiag_l1(error_ptm):
    nondiag_L1_opt_ideal = np.abs(error_ptm - np.diag(error_ptm.diagonal())).sum()
    return nondiag_L1_opt_ideal