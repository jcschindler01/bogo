
"""
Bosonic Bogoliubov transition amplitudes.

Hamiltonian
    H = (p.p + x.M.x)/2
with M a real symmetric classical coupling matrix and p,x are length N vectors.

a = local creation operators
b = intermediate creation operators
A = global creation operators

n = length N integer arrays of occupation numbers

(--> canonical ordering of n?)
"""

## import
import numpy as np


##### helper functions ######
#############################




############## amplitudes #################
###########################################

## local to intermediate
def ab(na,nb,M):
    """Calculate <na|nb> from local (a) to intermediate (b=Ra) modes."""
    return 0

## intermediate to global
def bA(nb,nA,M):
    """Calculate <na|nb> from local (a) to intermediate (b=Ra) modes."""
    return 0

## local to global
def ab(na,nA,M):
    """Calculate <na|nb> from local (a) to intermediate (b=Ra) modes."""
    return 0


