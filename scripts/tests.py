
## import
import numpy as np
import bogo as bg
from itertools import product as itprod
np.set_printoptions(precision=3, suppress=True)

##### tests #####
#################

def test1():
    """
    Test ab, the "orthogonal" part of transformation.
    This amplitude is total N conserving.
    It is derived by Wick's theorem.
    """
    ##
    print("TEST 1")
    print(test1.__doc__)
    ## params
    N = 3
    mu = .1
    na = np.array([1,0,4])
    nb = np.array([1,1,3])
    ## amp
    M = bg.V(N, mu)
    amp = bg.ab(na,nb,M)
    print("N, mu = %d, %f"%(N, mu))
    print("M =")
    print(M)
    print("Mvals = ")
    print(bg.eig(M)[0])
    print("Mvecs = ")
    print(bg.eig(M)[1])
    print()
    print("na, nb = ")
    print(na)
    print(nb)
    print()
    print("         ab = %8.3e"%amp)
    print("|<na|mb>|^2 = %8.3e"%(np.abs(amp)**2))
    print()

def test2():
    """
    Check the probability distribution of n,m vectors for ab.
    For fixed state <na| calculate all nonzero <na|nb> amplitudes.
    The state is either manual or random.

    Benchmarks:
    Time to calculate all nonzero amplitudes.
    (NxK = excitations x modes)
    6x6  = 2s       [na = np.array([6,0,0,0,0,0])]
    15x2 = 1s       [na = np.array([15,0])]
    7x7  = 15s      [na = np.array([7,0,0,0,0,0,0])]
    20x2 = 15s      [na = np.array([20,0])]

    Probably now limited by actualy python looping speed over 
    multiperms (n!/nb!..nb!) and over valid out states (?).

    """
    ##
    print("TEST 2")
    print(test2.__doc__)
    ## params
    mu = .2
    na = np.array([6,0,0,0,0,0])
    ## optional random
    if False:
        K = 6
        nmax = 3
        na = np.random.randint(0,nmax,K)
    ## derived
    K = len(na)
    N = int(np.sum(na))
    M = bg.V(K, mu)
    ##
    nbs = itprod(range(0,N+1), repeat=K)
    nbs = (x for x in nbs if np.sum(x)==N)
    summed = 0.0
    ##
    print("na = ", na)
    for nb in nbs:
        prob = np.abs(bg.ab(na,nb,M))**2
        summed += prob
        print(nb, "prob=% 8.6f, sum=% 5.3f"%(prob,summed))
    print("na = ", na)
    print()


def test3():
    """
    Test bA, the "squeezing" part of transformation.
    Each mode is individually squeezed.
    Not N preserving, but modes are decoupled.
    """
    ##
    print("TEST 3")
    print(test3.__doc__)
    ## params
    mu = 1
    nb = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    nA = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    ## amp
    K = len(nb)
    M = bg.V(K, mu)
    ##
    vals, vecs = bg.eig(M)
    w = np.sqrt(vals)
    r = np.arctanh((w-1)/(w+1))
    ##
    amp = bg.bA(nb,nA,M)
    ##
    print("K, mu = %d, %f"%(K, mu))
    print("M =")
    print(M)
    print("Mvecs = ")
    print(vecs)
    print("Mvals = ")
    print(vals)
    print("w = ")
    print(w)
    print("r = ")
    print(r)
    print()
    print("nb, nA = ")
    print(nb)
    print(nA)
    print()
    print("         bA = %8.3e"%(amp))
    print("|<nb|nA>|^2 = %8.3e"%(np.abs(amp)**2))
    print()



def test4():
    """
    For fixed state |nA> calculate <nb|nA> amplitudes
    for all <nb| with up to N+Ncut particles.
    Reports remainder = total prob of more than N+Ncut particles.

    Benchmarks:
    [5.2s] mu=.5, na=4,4,0,0, Ncutoff=8, remainder=2e-12
    [<1s ] mu=.5, na=0,0,0,0, Ncutoff=0, remainder=3e-2
    [<1s ] mu=.5, na=50, Ncutoff=60, remainder=6e-2

    Prints only probs greater than or equal to eps.
    """
    ##
    print("TEST 4")
    print(test4.__doc__)
    ## params
    mu = .5
    nA = np.array([50])
    Ncut = 10
    eps = 1e-6
    ## optional random
    if False:
        K = 6
        nmax = 3
        na = np.random.randint(0,nmax,K)
    ## derived
    K = len(nA)
    N = int(np.sum(nA))
    M = bg.V(K, mu)
    Ncutoff = N + Ncut
    ##
    nbs = itprod(range(0,Ncutoff+1), repeat=K)
    summed = 0.0
    ##
    print("nA = ", nA)
    for nb in nbs:
        prob = np.abs(bg.bA(nb,nA,M))**2
        summed += prob
        if prob>=eps:
            print(nb, "prob=% 8.6f, sum=% 5.3f"%(prob,summed))
    print("nA = ", nA)
    ##
    print("N = ", N)
    print("Ncutoff = ", Ncutoff)
    print("remainder = %8.3e"%(1.0-summed))
    print()



##### run #####
###############

if __name__=="__main__":
    test3()

