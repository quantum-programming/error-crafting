import math

from sage.all import QuadraticField
import itertools
import numpy as np

R = QuadraticField(2, "sqrt2")
zsqrt2_ring = R.ring_of_integers()
sqrt2 = zsqrt2_ring.gen(1)

def as_Zsqrt2_element(real_integer_coeff = 0, real_sqrt2_coeff = 0):
    return real_integer_coeff + real_sqrt2_coeff * sqrt2

class ZSqrt2:
    def __init__(self, a=0, b=0, z=None):
        if z is None:
            self.a = a
            self.b = b
            self.z = as_Zsqrt2_element(a, b)
        else:
            #print(z)
            self.a, self.b = z.list()
            self.z = z
                
    def decimal(self):
        return self.a + self.b * math.sqrt(2)

    def decimal_dot(self):
        return self.a - self.b * math.sqrt(2)
    
    def galois_conj(self):
        return ZSqrt2(self.a, -self.b)

    def norm(self):
        return self.z.norm()

    def dot(self):
        return ZSqrt2(self.a, -self.b)

    def __add__(self, other):
        if isinstance(other, ZSqrt2):
            return ZSqrt2(z = self.z + other.z)
        else:
            return ZSqrt2(z = self.z + other)

    def __sub__(self, other):
        if isinstance(other, ZSqrt2):
            return ZSqrt2(z = self.z - other.z)
        else:
            return ZSqrt2(z = self.z - other)

    def __mul__(self, other):
        if isinstance(other, ZSqrt2):
            return ZSqrt2(z = self.z * other.z)
        else:  # Assuming other is a scalar
            return ZSqrt2(z = self.z * other)

    def __truediv__(self, other):
        if isinstance(other, ZSqrt2):
            mag = other.norm()
            new_a = (self.a * other.a - 2 * self.b * other.b) / mag
            new_b = (self.b * other.a - self.a * other.b) / mag
            return ZSqrt2(new_a, new_b)
        else:
            return ZSqrt2(self.a/other, self.b/other)
        
    def mod(self, other):
        return self % other

    def __mod__(self, other):
        return ZSqrt2(z = self.z.mod(other.z))

    def __str__(self):
        return f"{self.a} + {self.b} sqrt2"
    def __repr__(self):
        return f"{self.__str__()}"
    
    def is_zero(self):
        return self.z.is_zero()
        
def get_gcd_ZSqrt2(a, b):
    if b.z.is_zero():
        return a
    else:
        return get_gcd_ZSqrt2(b, a%b)
    

#######################
# Zw
#######################

from sage.all import PolynomialRing, NumberField, QQ
import sympy
from sympy.ntheory import nextprime


# Define the polynomial w^4 + 1 and the ring Z[w]
P = PolynomialRing(QQ, 'w')
z = P.gen()
K = NumberField(z**4 + 1, 'w')
Zw_ring = K.ring_of_integers()
w = Zw_ring.gen(1)

def as_Zw_element(real_integer_coeff=0, real_sqrt2_coeff = 0, imag_integer_coeff = 0, imag_sqrt2_coeff = 0):
    a =(-real_sqrt2_coeff  + imag_sqrt2_coeff)
    b = imag_integer_coeff
    c = (real_sqrt2_coeff  + imag_sqrt2_coeff)
    d = real_integer_coeff
    return a * w**3 +  b * w**2 +  c* w + d 
    
def to_complex(element):
    ww = np.exp(1j * np.pi/4)
    return np.array([1, ww**1, ww**2, ww**3]) @ np.array(element.list())    

class Zw:
    def __init__(self, a=0, b=0, c=0, d=0, z=None):
        if z is None:
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.z = a * w**3 + b * w**2 + c * w + d
        else:
            #print(z)
            self.d, self.c, self.b, self.a = z.list()
            self.z = z
                
    def decimal(self):
        return self.z.n()
    
    def to_complex(self):
        ww = np.exp(1j * np.pi/4)
        return np.array([1, ww**1, ww**2, ww**3]) @ np.array([self.d, self.c, self.b, self.a])    

        #return to_complex(self.z)

    #def decimal_dot(self):
        #return self.a - self.b * math.sqrt(2)
    
    def galois_conj(self):
        return Zw(-self.a, self.b, -self.c, self.d)
        #return ZSqrt2(self.a, -self.b)

    def norm(self):
        a_, b_, c_, d_ = self.a, self.b, self.c, self.d
        return ZSqrt2(a_ * a_ + b_ * b_ + c_ * c_ + d_ * d_,
                      c_ * b_ + d_ * c_ + b_ * a_ - a_ * d_)

    #def dot(self):
        #return ZSqrt2(self.a, -self.b)


    def __pow__(self, power:int):
        z_buf = self.z
        z = self.z
        for _ in range(power-1):
            z_buf = z_buf * z
        return Zw(z=z_buf)

    def __add__(self, other):
        if isinstance(other, Zw):
            return Zw(z = self.z + other.z)
        else:
            return Zw(z = self.z + other)

    def __sub__(self, other):
        if isinstance(other, Zw):
            return Zw(z = self.z - other.z)
        else:
            return Zw(z = self.z - other)

    def __mul__(self, other):
        if isinstance(other, Zw):
            return Zw(z = self.z * other.z)
        else:  # Assuming other is a scalar
            return Zw(z = self.z * other)

    def __truediv__(self, other):
        if isinstance(other, Zw):
            #mag = other.norm()
            #new_a = (self.a * other.a - 2 * self.b * other.b) / mag
            #new_b = (self.b * other.a - self.a * other.b) / mag
            #return Zw(z = self.z / other.z)
            n = (self * other.conj()) * ((other * other.conj()).dot())
            mag = other.norm().norm()
            return Zw(int(np.floor(n.a / mag)), int(np.floor(n.b / mag)),
                    int(np.floor(n.c / mag)), int(np.floor(n.d / mag)))            
        else:
            return Zw(z = self. z / other)
        
    def mod(self, other):
        return self % other

    def __mod__(self, other):
        return Zw(z = self.z - (self.z / other.z) * other.z)
    
    def _unused__mod__(self, other):
        #ZOmega n = (Y * Z.conj()) * ((Z * Z.conj()).dot());
        n = (self * other.conj()) * ((other * other.conj()).dot())
        k = other.norm().norm()
        a1 = np.floor(n.a + (k // 2) / k)
        a2 = np.floor(n.b + (k // 2) / k)
        a3 = np.floor(n.c + (k // 2) / k)
        a4 = np.floor(n.d + (k // 2) / k)

        q = Zw((int(a1)), (int(a2)), (int(a3)), (int(a4)))

        return q * self - other
        #return Zw(z = self.z.mod(other.z))

    def __str__(self):
        return f"{self.a} w**3 + {self.b} w**2 + {self.c} w + {self.d}"
    def __repr__(self):
        return f"{self.__str__()}"
    
    def is_zero(self):
        return self.z.is_zero()
    
    def to_ZSqrt2(self):
        assert self.b==0
        return ZSqrt2(self.d, self.c)
    
    def conj(self):
        return Zw(-self.c, -self.b, -self.a, self.d)
    
    def dot(self):
        return Zw(-self.a, self.b, -self.c, self.d)
    
    
def get_gcd_Zw(a, b):
    if b.z.is_zero():
        return a
    else:
        return get_gcd_Zw(b, a%b)        