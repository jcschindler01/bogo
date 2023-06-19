
## import
import numpy as np
import bogo as bg
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
    na = np.array([2,0,3])
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
    """
    ##
    print("TEST 2")
    print(test1.__doc__)
    ## params
    K = 3
    mu = .2
    na = np.random.randint(0,4,K)
    ## coupling
    M = bg.V(K, mu)
    ##
    print("na = ", na)
    ##
    N = int(np.sum(na))
    summed = 0
    for nb1 in range(N+1)[::-1]:
        for nb2 in range(N+1-nb1)[::-1]:
            nb = np.array([nb1,nb2,N-nb1-nb2])
            prob = np.abs(bg.ab(na,nb,M))**2
            summed += prob
            print(nb, "prob=% 5.3f, sum=% 5.3f"%(prob,summed))



##### run #####
###############

if __name__=="__main__":
    test2()

