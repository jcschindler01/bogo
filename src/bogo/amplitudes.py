
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
import itertools as it
from collections import Counter
from scipy.special import factorial as fac


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
    RR = R[1]
    
    # Convert the states to set of how many operators
    sa = np.repeat(np.arange(len(na)), na)
    sb = np.repeat(np.arange(len(nb)), nb)

    if len(sa)!=len(sb):
      return 0.
    
    # permutations of the first set
    perm_sa = np.array(list(it.permutations(sa)))
    counts = Counter(tuple(array) for array in perm_sa)
    # unique permutations
    unique_perms = [np.array(arr) for arr in counts.keys()]
    # how many times does each unique permutation appear?
    counts = list(counts.values())
    
    facsa = np.array([fac(nai) for nai in na])
    facsb = np.array([fac(nbi) for nbi in nb])
    prefactor = 1/np.sqrt(np.prod(facsa)*np.prod(facsb))
    
    sum_result = 0
    
    for k in range(len(unique_perms)):
        r = 1
        for i in range(len(unique_perms[k])):
            r *= RR[sb[i]][unique_perms[k][i]]
        sum_result += counts[k]*r

    return prefactor*sum_result

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


