
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
    print(repr(amp))
    return None
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
    mu = .1
    na = na = np.array([6,0,0,0,0,0])
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



##### run #####
###############

if __name__=="__main__":
    test2()

