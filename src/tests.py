
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
    na = np.array([9,0,0])
    nb = np.array([1,1,7])
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



##### run #####
###############

if __name__=="__main__":
    test1()

