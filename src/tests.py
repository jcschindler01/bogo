
## import
import numpy as np
import bogo as bg

## classical coupling matrix
def Mpot(mu):
    return np.array([[1, -mu],[-mu, 1]])

##### tests #####
#################

def test1():
    """
    Test ab, the "orthogonal" part of transformation.
    """
    ## params
    mu = 0.1
    na = np.array([3,0])
    nb = np.array([2,1])
    ## amp
    M = Mpot(mu)
    amp = bg.ab(na,nb,M)
    print(amp)



##### run #####
###############

if __name__=="__main__":
    test1()


# ## coupling
# mu = 0.25
# M = np.array([[1, -mu],[-mu, 1]])

# ## occupation numbers
# na = np.array([0,0])
# nb = np.array([1,2])
# # nA = np.array([1,2])

# ## calculate
# ab = bg.ab(na,nb,M=M)
# # bA = bg.bA(nb,nA,M=M)
# # aA = bg.aA(na,nA,M=M)


# ## results
# print()
# print("M =")
# print(M)
# print()
# print("na = %s"%(na))
# print("nb = %s"%(nb))
# # print("nA = %s"%(na))
# print()
# print(" <na|nb>    = %.3f"%(ab))
# print("|<na|nb>|^2 = %.3f"%(np.abs(ab)**2))
# print()

