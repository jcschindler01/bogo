
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
from multipermute import permutations as mperm
# from sympy.utilities.iterables import multiset_premutations as mperm


##### helper functions ######
#############################

def n_subspace(N,n):
    '''Return states of length N whose elements add up to n.'''
    states = np.array(list(it.product(np.arange(n+1),repeat=N)))
    states_in_subspace = []
    
    for state in states:
        if np.sum(state)==n:
            states_in_subspace.append(state)
    subspace = np.array(states_in_subspace)
    
    return subspace 

def single(n,m,eta):
    if eta<0:
        n,m = m,n
        eta = -eta
    if (n%2)==(m%2):
        Min = min(n,m)
        if n%2 == 0:
            K = np.arange(0,Min+1,2)
        if n%2 ==1:
            K = np.arange(1,Min+1,2)
        prefactor = (np.sqrt(fac(m)*fac(n)))/((np.cosh(eta))**((n+m+1)/2))
        terms = [(((np.sinh(eta))/2)**((n+m-2*k)/2))*(((-1)**((n-k)/2))/(fac(k)*fac(int((n-k)/2))*fac(int((m-k)/2)))) for k in K]
        summ = np.sum(np.array(terms))
        inner_single = prefactor*summ
        return inner_single
    else:
        return 0.

def fast_single(n,m,eta):
    if np.abs(n-m)>=20:
        return 0.
    else:
        return single(n,m,eta)
    


############## amplitudes #################
###########################################

## local to intermediate
def ab(na,nb,M):
    """Calculate <na|nb> from local (a) to intermediate (b=Ra) modes."""
    vals, vecs = np.linalg.eig(M)
    sorted_indices = np.argsort(vals)
    RR = vecs[:, sorted_indices].T

    # Convert the states to set of how many operators
    sa = np.repeat(np.arange(len(na)), na)
    sb = np.repeat(np.arange(len(nb)), nb)

    if len(sa)!=len(sb):
        return 0.

    
    # version 1:
#     # permutations of the first set
#     perm_sa = np.array(list(it.permutations(sa)))
#     counts = Counter(tuple(array) for array in perm_sa)
#     # unique permutations
#     unique_perms = [np.array(arr) for arr in counts.keys()]
#     # how many times does each unique permutation appear?
#     counts = list(counts.values())

    # version 2:
#     counts = Counter()

#     for perm in it.permutations(sa):
#         counts[tuple(perm)] += 1

#     unique_perms = np.array(list(counts.keys()))
#     counts = list(counts.values())
#     print('counts=')
#     print(counts)

    if len(sa)==len(sb):
        if len(sa)==0:
            return 1.
        else:
            facsa = np.array([fac(nai) for nai in na])
            facsb = np.array([fac(nbi) for nbi in nb])
            prefactor = 1/np.sqrt(np.prod(facsa)*np.prod(facsb))
    
    
#     print('np.prod(fac(nb))=')
#     print(np.prod(fac(nb)))
    
            sum_result = 0
    
    
    #version 1 (the one we worked with in the meeting):
#     for k in range(len(unique_perms)):
#         r = 1
#         for i in range(len(unique_perms[k])):
#             r *= RR[sb[i]][unique_perms[k][i]]
#         sum_result += counts[k]*r

    #version 2:
#     for i,uni in enumerate(unique_perms):
#         r = 1
#         for b,u in zip(sb,uni):
#             r *= RR[b,u]
#         sum_result += counts[i]*r 
    
    #version 3:
#     for i,uni in enumerate(unique_perms):
#         sum_result += counts[i]*np.prod(np.array([RR[b,u] for b,u in zip(sb,uni)]))


    # version multipermute
            sym_a = np.prod(fac(na))
            sym_b = np.prod(fac(nb))

            if sym_b<=sym_a:
                for bf in mperm(sb):
                    sum_result += sym_b*np.prod([RR[sa[z],bf[z]] for z in range(np.sum(na))])
            else:
                for af in mperm(sa):
                    sum_result += sym_a*np.prod([RR[af[z],sb[z]] for z in range(np.sum(na))])
    
    
#     for bf in mperm(sb):
#            sum_result += sym_b*np.prod([RR[sa[z],bf[z]] for z in range(np.sum(na))])   
    
    
    

            return prefactor*sum_result

## intermediate to global
def bA(nb,nA,M):
    """Calculate <nb|nA> from intermediate (b) to global (A) modes."""
    
    vals, vecs = np.linalg.eig(M)
    sorted_indices = np.argsort(vals)
    R = vals[sorted_indices]   
    Omega = np.sqrt(R)
    Eta = np.arctanh((Omega-1)/(Omega+1))
    inner = 1
    for i in range(len(nA)):
        inner *= single(nA[i],nb[i],Eta[i])
    
    return inner

## Faster version of bA:
def fast_bA(nb,nA,M):
    vals, vecs = np.linalg.eig(M)
    sorted_indices = np.argsort(vals)
    R = vals[sorted_indices]   
    Omega = np.sqrt(R)
    Eta = np.arctanh((Omega-1)/(Omega+1))
    inner = 1
    for i in range(len(nA)):
        inner *= fast_single(nA[i],nb[i],Eta[i])
    
    return inner
    


## local to global
def aA(na,nA,M):
    """Calculate <na|nb> from local (a) to global (A) modes."""
    N = np.sum(na)
    l = len(na)
    
    states = np.array(list(it.product(np.arange(N+1),repeat=l)))
    states_in_subspace = []
    
    for state in states:
        if np.sum(state)==N:
            states_in_subspace.append(state)
    subspace = np.array(states_in_subspace)
    
    f = np.array([ab(na,sb,M) for sb in subspace])
    s = np.array([bA(sb,nA,M) for sb in subspace])

    return np.dot(f,s)

## Fast version of aA:
def fast_aA(na,nA,M):
    N = np.sum(na)
    l = len(na)
    
    states = np.array(list(it.product(np.arange(N+1),repeat=l)))
    states_in_subspace = []
    
    for state in states:
        if np.sum(state)==N:
            states_in_subspace.append(state)
    subspace = np.array(states_in_subspace)
    
    f = np.array([ab(na,sb,M) for sb in subspace])
    s = np.array([fast_bA(sb,nA,M) for sb in subspace])

    return np.dot(f,s)
    


