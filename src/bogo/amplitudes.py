
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


def squeeze(n,m,r):
    """
    Single mode squeezing <n|m,r> as in quant-ph/0108024 equation (20).
    """
    ##
    amp = 0.0

    ## zero if no parity
    if n%2 != m%2:
        return 0.0

    ## range to sum
    krange = range(n%2, min(n,m)+1, 2)

    ## prefactor
    Z = np.sqrt(fac(m)*fac(n)) / np.cosh(r)**((n+m+1)/2)

    ## sum over k
    for k in krange:
        A = (np.sinh(r)/2)**((n+m-2*k)/2)
        B = (-1)**((n-k)/2)
        C = fac(k)*fac((m-k)/2)*fac((n-k)/2)
        amp += Z * A*B/C

    ## return
    return amp


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
def bA(nb,nA,M):
    """
    Calculate <nb|nA> from intermediate (b) to global (A, squeezed) modes.

    Each mode is squeezed individually, <nb|nA>=<nb0|nA0> *...* <nbK|nAK>.
    Squeezed number states related as in quant-ph/0108024 equation (20).
    Not N preserving but modes are decoupled.

    Calculate as if nb and nA were single values, not arrays,
    just take product at the end.

    Squeezing factor r ranges for our system
    from -infty at w=0 to arctanh(1/3)=.35 at w=2.

    Mode ordering must be same (sorted freq) as in ab function.
    """
    ##
    K = len(nb)
    w = np.sqrt(eig(M)[0])
    
    ##
    r = np.arctanh((w-1)/(w+1))
    amps = np.array([squeeze(nb[k],nA[k],r[k]) for k in range(K)])
    amp = np.prod(amps)

    ## return
    return amp


## local to global
def aA(na,nA,R):
    """Calculate <na|nb> from local (a) to global (A) modes."""
    ##
    amp = 0.0
    ## return
    return amp
