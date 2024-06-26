
import itertools
import sympy
import math


import numpy as np

from UnifiedCompiler.coherent.ring import ZSqrt2, Zw

def _synthesize_Rz_rus(angle, eps, verbose = 0, q = 0.9, max_success_count=3, as_matrix = True, use_real_number = True):
    pass
    #return sth

def synthesize_Rz_rus(angle, eps, verbose = 0, q = 0.9, max_success_count=3, as_matrix = True, use_real_number = False):
    assert as_matrix, "gate decomposition not implemented yet."

    ww = np.exp(1j * np.pi/4)
    z, z_coeffs = approximate_rotation_angle_by_cyclotomic_rational(angle, eps)
    z_gc = (-1) * z_coeffs[0] * ww**3 + z_coeffs[1] * ww**2 + (-1) * z_coeffs[2] * ww + z_coeffs[3]
    unitary = np.array([[1, 0], [0, np.conjugate(z)/z]])
    
    success = False
    N_t = np.inf
    success_count = 0
    ret = [np.NaN, np.NaN, np.NaN, np.NaN]
    for ra in range(1, 20):
        for rb in range(ra):
            r = ra + rb * np.sqrt(2)
            r_gc = ra - rb * np.sqrt(2)
            Lr_min = int(np.ceil(np.log2(np.abs(z)**2 * r**2)))//2
            Lr_max = int(np.ceil(np.log2(np.abs(z)**2 * r**2))) + 3


            for Lr in range(Lr_min, Lr_max):
                if 2**Lr <= np.abs(r_gc * z_gc)**2:
                    continue

                #xi = (Zw(*z_coeffs) * (as_Zw_element(ra, rb))).norm() * (-1) + 2**(Lr)
                #xi = Zw(*z_coeffs).norm() * (ZSqrt2(ra, rb)*ZSqrt2(ra, rb)) * (-1) + 2**Lr
                a_, b_, c_, d_ = z_coeffs
                znorm_a = a_ * a_ + b_ * b_ + c_ * c_ + d_ * d_
                znorm_b = c_ * b_ + d_ * c_ + b_ * a_ - a_ * d_

                rsq_a = ra * ra  + 2 * rb * rb
                rsq_b = ra * rb + rb * ra

                xi_a = 2**Lr - (znorm_a * rsq_a + 2 * znorm_b * rsq_b)
                xi_b = -(znorm_b * rsq_a + znorm_a * rsq_b)
                #xi = ZSqrt2(xi_a, xi_b)
                prob = 1 - (xi_a + xi_b * np.sqrt(2))/2**(Lr)
                if prob <0 or prob > 1:
                    continue
                solved = is_easily_solvable(xi_a, xi_b)
                success = solved and prob>=q
                if success:
                    success_count += 1
                    n_t = 2*Lr/prob
                    if N_t > n_t:
                        N_t = n_t
                        
                        ret = [unitary, z, N_t, prob, Lr, ra, rb, xi_a, xi_b, z_coeffs]
                    if verbose:
                        print(f"success at {(ra, rb, Lr)=}")
                if success_count >= max_success_count:
                    break
            if success_count >= max_success_count:
                break
        if success_count >= max_success_count:
            break            
            
    return ret

def approximate_rotation_angle_by_cyclotomic_rational(theta, eps, return_coeffs = False):
    """
    returns z such that |z^*/z - np.exp(1j * theta)| < eps.
    args:
        theta: rotation angle of pauli Z.
        eps: accuracy
    output:
        z : some complex number
        z_coeff = [a, b, c, d] that yields z = a * w**3 + b * w**2 + c * w**1 + d * w**0
    """
    from mpmath import pslq
    w = np.exp(1j * np.pi/4)
    a, b, c, d = pslq([np.cos(theta/2) -np.sin(theta/2), np.sqrt(2) * np.cos(theta/2), np.cos(theta/2) + np.sin(theta/2), np.sqrt(2) * np.sin(theta/2)], tol = eps, maxcoeff=10000)
    if np.abs(np.inner([a, b, c, d], [w**3, w**2, w**1, w**0]) - np.exp(1j*theta)) > np.abs(np.inner([-a, -b, -c, -d], [w**3, w**2, w**1, w**0]) - np.exp(1j*theta)):
        a = -a
        b = -b
        c = -c
        d = -d    
    coeffs = [a, b, c, d]
    z = a * w**3 + b * w**2 + c * w**1 + d * w**0
    if return_coeffs:
        return z, coeffs
    return z, coeffs

def is_integer_norm_equation_easily_solvable(integer):
    res = sympy.ntheory.factorint(integer)
    
    for p, power in res.items():
        if power %2 == 0 or p%8 in [1,2, 3, 5, 7]:
            continue
        else:
            return False
    return True

def is_Zsqrt2_norm_equation_easily_solvable(xii_a, xii_b):
    p_prod = int(xii_a**2 - 2 * (xii_b**2))
    res = sympy.ntheory.factorint(int(p_prod))
    
    for p, power in res.items():
        if power %4 == 0 or p%8 in [1,2]:
            continue
        else:
            return False
    return True
    

def is_easily_solvable(xi_a, xi_b):
    if xi_a + xi_b * np.sqrt(2) < 0:
        return False
    
    #xi = zsqrt2_ring(xi) # to assert input to be sage object
    #a, b = xi.list()
    gcd = math.gcd(xi_a, xi_b)
    xii_a, xii_b = xi_a//gcd, xi_b//gcd
    
    
    # step 1 gcd of 
    #T1, sucess = solve_norm_equation_for_integer(gcd)
    success = is_integer_norm_equation_easily_solvable(gcd)
    if not success:
        return False
    
    # step 2 factor by 
    success = is_Zsqrt2_norm_equation_easily_solvable(xii_a, xii_b)
    if not success:
        return False    

    return True

