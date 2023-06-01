
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

# eigendecomposition of M (we don't want to include this process in each function!)
def R(M):
    vals, vecs = np.linalg.eig(M)
    sorted_indices = np.argsort(vals)
    sorted_eigenvalues = vals[sorted_indices]
    sorted_eigenvectors = vecs[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors.T
    




############## amplitudes #################
###########################################

## local to intermediate
def ab(na,nb,R):
    """Calculate <na|nb> from local (a) to intermediate (b=Ra) modes."""
    return 0

## intermediate to global
def bA(nb,nA,R):
    """Calculate <nb|nA> from intermediate (b) to global (A) modes."""
    return 0

## local to global
def ab(na,nA,R):
    """Calculate <na|nb> from local (a) to global (A) modes."""
    return 0


