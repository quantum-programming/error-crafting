import math

from UnifiedCompiler.coherent.ring import as_Zw_element, get_gcd_Zw, Zw

from sage.all import PolynomialRing, NumberField, QQ
import sympy
from sympy.ntheory import nextprime

#######################
# Step 1
#######################

# Define the polynomial w^4 + 1 and the ring Z[w]
P = PolynomialRing(QQ, 'w')
z = P.gen()
K = NumberField(z**4 + 1, 'w')
Zw_ring = K.ring_of_integers()
w = Zw_ring.gen(1)

#def as_Zw_element(real_integer_coeff=0, real_sqrt2_coeff = 0, imag_integer_coeff = 0, imag_sqrt2_coeff = 0):
    #a =(-real_sqrt2_coeff  + imag_sqrt2_coeff)
    #b = imag_integer_coeff
    #c = (real_sqrt2_coeff  + imag_sqrt2_coeff)
    #d = real_integer_coeff
    #return a * w**3 +  b * w**2 +  c* w + d 
    
def to_complex(element):
    ww = np.exp(1j * np.pi/4)
    return np.array([1, ww**1, ww**2, ww**3]) @ np.array(element.list())

# Function to calculate gcd in Z[w]
def gcd_in_Zw(x, y):
    return Zw_ring(x).gcd(Zw_ring(y))

def is_good_integer(integer):
    res = sympy.ntheory.factorint(integer)
    for p, power in res.items():
        if power %2 == 1 and p%8 == 7:
            return False
    return True

def is_divisible(target, s):
    """
    returns if target % s ==0.
    """
    if isinstance(target, int):
        return is_integer(target/s)
    return all([coef.is_integer() for coef in (target/s).list()])

def get_gcd_in_Zw(p):
    coeff = 1
    while True:
        u = np.sqrt(coeff * p - 1)
        if is_integer(u):
            break
        else:
            coeff += 1
    u_plus_i = as_Zw_element(int(u), imag_integer_coeff=1)

    a = Zw(*as_Zw_element(*tmp.list()).list())
    b = Zw(*u_plus_i.list())
    t = get_gcd_Zw(a, b)

    return t

def solve_norm_equation_for_integer(integer):
    """
    returns the solution of norm equation 
       t^dag t = integer
    for t in Z[w] where w is root of w^4 + 1.

    argss:
        integer: integer that gives the norm
    output:
        t: root that satisfy t^dag t sim integer
        sucess: tells you if solution is found.    

    We have possibly three situations for output (T, sucess) where
      sucess   = True    → solution found and t^dag t sim integer satisfied.
      success = False  → not easily solvable
    """
    assert is_integer(integer), f"input {integer} is not an integer."
    res = sympy.ntheory.factorint(integer)
    
    T = as_Zw_element(1)
    for p, power in res.items():
        if power %2 == 0:
            T *= (p**(power//2))
        elif p %8 != 7:
            t = find_t_for_rational_prime(p)
            T *= (t**power)
        else:
            return None, False
        
    success = np.isclose(integer, to_complex(T.conjugate() * T))
    # is there any situation that T.conj() * T != integer?
    #T *= np.sqrt(integer / )
    return T, True

def is_integer(number):
    
    if int(number) == number:
        return True
    if np.isclose(int(number), number, rtol = 1e-15, atol = 1e-15):
        return True
    return False

def find_t_for_rational_prime(xii):
    """
    find t in Z[w] such that t^dag t sim xii.
    Here, sim denotes that two numbers are equivalent up to the unit of the ring Z[w].
    """
    assert sympy.isprime(xii)
    if xii%8 in [1, 5]:
        coeff = 1
        while True:
            root = np.sqrt(coeff * xii - 1)
            if is_integer(root):
                break
            else:
                coeff += 1
        u_plus_i = as_Zw_element(int(root), imag_integer_coeff=1)
        t = gcd_in_Zw(xii, u_plus_i)
    elif xii % 8 == 3:
        coeff = 1
        while True:
            root = np.sqrt(coeff * xii - 2)
            if is_integer(root):
                break
            else:
                coeff += 1
        u_plus_isqrt2 = as_Zw_element(int(root), imag_sqrt2_coeff=1)
        t = gcd_in_Zw(xii, u_plus_isqrt2)
    elif xii%8 == 7:
        return None
    elif xii == 2:
        return as_Zw_element(real_sqrt2_coeff=1)
    return t

#######################
# Step 2
#######################
from sage.all import QuadraticField
def get_product(lis):
    if not isinstance(lis, list):
        return lis
    t = lis[0]
    for _t in lis[1:]:
        t *= _t
    return t

R = QuadraticField(2, "sqrt2")
zsqrt2_ring = R.ring_of_integers()
sqrt2 = zsqrt2_ring.gen(1)

def as_Zsqrt2_element(real_integer_coeff = 0, real_sqrt2_coeff = 0):
    return real_integer_coeff + real_sqrt2_coeff * sqrt2

def to_real(Zsqrt2_element):
    return Zsqrt2_element.n()

def gcd_in_Zsqrt2(x, y):
    return zsqrt2_ring(x).gcd(zsqrt2_ring(y))

#def is_integer(x):
    #return np.isclose(x, int(x), atol = 1e-10)


def get_root_under_Zsqrt2(p):
    """
    Follows Lemma C.12 of Ross & Selinger, arXiv:1403.2975.
    """
    
    assert sympy.isprime(p) and p%8 == 1
    
    count = 1
    while True:
        x = np.sqrt(count * p + 2)
        if is_integer(x):
            return gcd_in_Zsqrt2(p, int(x) + sqrt2)
        count += 1

import sympy
import numpy as np
import itertools
def solve_norm_equation_for_Zsqrt2(xi):
    xi = zsqrt2_ring(xi)
    
    p_prod = xi.galois_conjugate() * xi
    if p_prod == 1 or p_prod < 0:
        return None, False
    
    res = sympy.ntheory.factorint(int(p_prod))
    
    t_list = []
    power_list = []
    
    #T = as_Zw_element(1)
    for p, power in res.items():
        if power %4 == 0:
            t = p**(power//4)
            
            t_list.append(t)
            power_list.append(power)
            
        elif p%8 in [1, ]:
            tmp = get_root_under_Zsqrt2(p)
            
            coeff = 1
            while True:
                u = np.sqrt(coeff * p - 1)
                if is_integer(u):
                    break
                else:
                    coeff += 1
            u_plus_i = as_Zw_element(int(u), imag_integer_coeff=1)
            t = gcd_in_Zw(as_Zw_element(*tmp.list()), u_plus_i)
            
            t_list.append(t)
            power_list.append(power)
            
        elif p%8 in [3, 5, 7]:
            return None, False
        elif p == 2:            
            t = 1 + w
            
            t_list.append(t)
            power_list.append(power)
            
        #T *= (t**power)
    
    # possibly you are computing the other one
    success = False
    for b_list in itertools.product([0, 1, 2, 3], repeat = len(t_list)):
        T = get_product([t_list[j].galois_conjugates(Zw_ring)[b_list[j]] **power_list[j] for j in range(len(t_list))])
        if is_divisible(xi, Zw_to_Zsqrt2(T.conjugate() * T)):
            #return T_gconj, True
            #print(f"{T=} is good")
            success = True
            break
    if not success:
        raise Exception("something wrong")
    
    # resolve unit ambiguity, since step 2  only assures t^\dag t \sim xi
    Tdag_T = Zw_to_Zsqrt2(T.conjugate() * T)
    coeff = xi/Tdag_T # this is (-1)**n * (lambd)**m

    if coeff == 1:
        return T, True
    elif coeff == -1:
        T= as_Zw_element(coeff).sqrt() * T
        return  T, True
    else:
        #print(f"{coeff=}")
        #print(f"{coeff.sqrt()=}")
        #print(f"{(coeff.sqrt()).list()=}")
        
              
        T = as_Zw_element(*(coeff.sqrt()).list()) * T
    
    return T, True
        
def Zw_to_Zsqrt2(element):
    assert element.list()[1] + element.list()[3] == 0, "this does not belong to Zsqrt2."
    assert element.list()[2] == 0, "this does not belong to Zsqrt2."
    
    # d + c*w + b * w**2 + a * w**3
    d, c, b, a = element.list()
    return (d) + (c) * sqrt2    

def solve_norm_equation(xi):
    if xi < 0:
        return None, False
    
    xi = zsqrt2_ring(xi) # to assert input to be sage object
    a, b = xi.list()
    gcd = math.gcd(a, b)
    xii = xi//gcd
    
    
    # step 1 gcd of 
    T1, sucess = solve_norm_equation_for_integer(gcd)
    if not sucess:
        return None, False
    
    # step 2 factor by 
    T2, success = solve_norm_equation_for_Zsqrt2(xii)
    if not success:
        return None, False
    
    # resolve unit ambiguity, since step 2  only assures t^\dag t \sim xi
    T_tmp = T1 * T2
    Tdag_T = Zw_to_Zsqrt2(T_tmp.conjugate() * T_tmp)
    coeff = xi/Tdag_T # this is (-1)**n * (lambd)**m

    T = as_Zw_element(*(coeff.sqrt()).list()) * T_tmp    
    
    return T, True