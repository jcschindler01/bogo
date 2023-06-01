
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
import intertools


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
    
    sa = np.repeat(np.arange(len(na)) + 1, na)
    sb = np.repeat(np.arange(len(nb)) + 1, nb)
    
    perm_sa = np.array(list(itertools.permutations(sa)))
    
    
    
    return 0

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
    for i in range(len(na)):
        inner *= single(na[i],nb[i],R,i)
    
    return inner


## local to global
def ab(na,nA,R):
    """Calculate <na|nb> from local (a) to global (A) modes."""
    return 0


