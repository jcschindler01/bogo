
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
from scipy.special import factorial as fac

from itertools import permutations as perm
from sympy.utilities.iterables import multiset_permutations as mperm

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
        print(bf)
        print([(afock[z],bf[z]) for z in range(N)])
        print(np.array([R[afock[z],bf[z]] for z in range(N)]))
        print(sym)
        print(sym * np.prod([R[afock[z],bf[z]] for z in range(N)]))
        amp += sym * np.prod([R[afock[z],bf[z]] for z in range(N)])

    ## more efficient using combos?
    # indices = range(N)
    # vals = np.zeros(len(N))
    # for idA in combo(indices, na[0]):
    #     indices.pop()

    ## return
    return Z*amp

## intermediate to global
def bA(nb,nA,R):
    """Calculate <nb|nA> from intermediate (b) to global (A) modes."""
    Omega = np.sqrt(R[0])
    Eta = np.arctanh((Omega-1)/(Omega+1))
    
    def single(n,m,R,i):
        omega = Omega[i]
        eta = Eta[i]
        
        if (n%2)==(m%2):
            Min = min(n,m)
            if n%2 == 0:
                K = np.arange(0,Min+1,2)
            if n%2 ==1:
                K = np.arange(1,Min+1,2)
            prefactor = (np.sqrt(fac(m)*fac(n)))/((np.cosh(eta))**((n+m+1)/2))
            terms = np.array([(((np.sinh(eta))/2)**((n+m-2*k)/2))*(((-1)**((n-k)/2))/(fac(k)*fac(int((n-k)/2))*fac(int((m-k)/2)))) for k in K])
            sum = np.sum(terms)
            inner_single = prefactor*sum
            return inner_single
        else:
            return 0.
   
    inner = 1
    for i in range(len(nA)):
        inner *= single(nA[i],nb[i],R,i)
    
    return inner


## local to global
def aA(na,nA,R):
    """Calculate <na|nb> from local (a) to global (A) modes."""
    N = np.sum(na)
    l = len(na)
    
    states = np.array(list(it.product(np.arange(N+1),repeat=l)))
    states_in_subspace = []
    
    for state in states:
        if np.sum(state)==N:
            states_in_subspace.append(state)
    subspace = np.array(states_in_subspace)
    
    f = np.array([ab(na,sb,R) for sb in subspace])
    s = np.array([bA(sb,nA,R) for sb in subspace])

    return np.dot(f,s)


