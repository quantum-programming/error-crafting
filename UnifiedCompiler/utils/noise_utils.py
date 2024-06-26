import numpy as np
import scipy
import matplotlib.pyplot as plt
#%matplotlib inline

from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge
from qulacs.gate import AmplitudeDampingNoise
#from qulacs.gate import BitFlipNoise, DephasingNoise
from qulacs import DensityMatrix
import qulacs
from qulacs.gate import SWAP
from qulacs.gate import Probabilistic, Identity, X, Y, Z

####################################
# Noise channels
####################################

def MyDepolarizingNoise(idx, pdep):
    return Probabilistic([1 - 3*pdep/4, pdep/4, pdep/4, pdep/4], [Identity(idx), X(idx), Y(idx), Z(idx)])

def MyDephasingNoise(idx, pdeph):
    return Probabilistic([1 - pdeph, pdeph], [Identity(idx), Z(idx)])

def InverseDepolarizingNoise(idx, pdep):
    pdepinv = (pdep/(1 - pdep))
    return MyDepolarizingNoise(idx, -pdepinv)

def MyGlobalDepolarizingNoise(n_qubit, pdep):
    pauli_list = [Identity(0), X(0), Y(0), Z(0)]
    for i in range(1, n_qubit):
        pauli_list = [merge(P(i), _Q) for P in [Identity, X, Y, Z] for _Q in pauli_list]
    prob_list = [1 - (4**n_qubit - 1)*pdep/(4**n_qubit)] + [pdep/4**n_qubit for _ in range(4**n_qubit-1)]
    return Probabilistic(prob_list, pauli_list)

def MyInverseGlobalDepolarizingNoise(n_qubit, pdep):
    pdepinv = pdep/(1-pdep)
    return MyGlobalDepolarizingNoise(n_qubit, -pdepinv)    

def MyTwoQubitDepolarizingNoise(i, j, pdep):
    return Probabilistic([1 - 15*pdep/16] +  [pdep/16]* 15, [merge(Pi(i), Pj(j)) for Pi in [Identity, X, Y, Z] for Pj in [Identity, X, Y, Z]])


def InverseTwoQubitDepolarizingNoise(i, j, pdep):
    pdepinv = (pdep/(1 - pdep))
    #pdepinv = pdep
    return MyTwoQubitDepolarizingNoise(i, j, -pdepinv)