

import numpy as np
import bogo as bg


## coupling
mu = 0.25
M = np.array([[1, -mu],[-mu, 1]])

## occupation numbers
na = np.array([0,0])
nb = np.array([1,2])
# nA = np.array([1,2])

## calculate
ab = bg.ab(na,nb,M=M)
# bA = bg.bA(nb,nA,M=M)
# aA = bg.aA(na,nA,M=M)


## results
print()
print("M =")
print(M)
print()
print("na = %s"%(na))
print("nb = %s"%(nb))
# print("nA = %s"%(na))
print()
print(" <na|nb>    = %.3f"%(ab))
print("|<na|nb>|^2 = %.3f"%(np.abs(ab)**2))
print()

