
"""
Bosonic Bogoliubov transition amplitudes.

Hamiltonian
    H = (p.p + x.M.x)/2
with M a real symmetric classical coupling matrix and p,x are length N vectors.

a = local creation operators
b = intermediate creation operators
A = global creation operators

n = length N integer arrays of occupation numbers

(Global n ordered from low frequency (left/0) to high freq (right/N))
"""

## import
import numpy as np
from scipy.special import factorial as fac
#from sympy.utilities.iterables import multiset_permutations as mperm
from bogo.multipermute import permutations as mperm



##### helper functions ######
#############################

def eig(M):
    """ Sorted eigendecomposition."""
    vals, vecs = np.linalg.eig(M)
    sort_idx = np.argsort(vals)
    sort_vals = vals[sort_idx]
    sort_vecs = vecs[:, sort_idx]
    return sort_vals, sort_vecs.T


############## amplitudes #################
###########################################

## local to intermediate
def ab(na,nb,M):
    """
    Calculate <na|nb> from local (a) to intermediate (b=Ra) modes.
    
    amp = sum_{complete wick contractions} R_{i,j}*...*R_{i,j}
    """
    ## initialize amp
    amp = 0.0 + 0.0j

    ## total particle number
    N = np.sum(na)

    ## zero unless N=M
    if not np.sum(nb)==N:
        return amp

    ## number rep to fock rep
    afock = np.repeat(range(len(na)), na)
    bfock = np.repeat(range(len(nb)), nb)

    ## orthogonal R diagonalizing classical M
    R = eig(M)[1]

    ## prefactor
    Z = 1./np.sqrt(np.prod(fac(na)*fac(nb)))

    ## multipermution symmetry factor (perms = sym*mperms)
    sym = np.prod(fac(nb))

    ## iterate over bfock permutations
    for bf in mperm(bfock):
        amp += sym * np.prod([R[afock[z],bf[z]] for z in range(N)])

    ## return
    return Z*amp


## intermediate to global
def bA(nb,nA,R):
    """Calculate <nb|nA> from intermediate (b) to global (A) modes."""
    ##
    amp = 0.0 + 0.0j
    ## return
    return amp


## local to global
def aA(na,nA,R):
    """Calculate <na|nb> from local (a) to global (A) modes."""
    ##
    amp = 0.0 + 0.0j
    ## return
    return amp
