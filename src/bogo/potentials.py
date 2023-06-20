
import numpy as np


"""
Classical coupling matrices with various boundary conditions.
"""

######################## matrix builders #################

def diag_ones(N):
    """
    N x N matrix with ones on the diagonal.
    """
    return np.diag(np.ones(N))

def offd_ones(N):
    """
    N x N matrix with ones above and below the diagonal.
    """
    ## initialize
    a = np.zeros((N,N))
    ## fill upper
    for i in range(N):
        for j in range(N):
            if i==j-1:
                a[i,j] = 1.
    ## fill lower
    for i in range(N):
        for j in range(N):
            if i-1==j:
                a[i,j] = 1.
    ## return
    return a

def corner_ones(N):
    """
    N x N matrix with ones in top right and bottom left.
    """
    ## initialize
    a = np.zeros((N,N))
    ## fill values
    a[0,N-1] = 1.
    a[N-1,0] = 1.
    ## return
    return a

def firstlast_ones(N):
    """
    N x N matrix with ones in first and last diagonal elements.
    """
    ## initialize
    a = np.zeros((N,N))
    ## fill values
    a[0,0] = 1.
    a[N-1,N-1] = 1.
    ## return
    return a

##############################################################

############## potential builders ###############################

def Vij_closed(N, mu=0.2):
    """
    Potential matrix with closed boundary conditions (i.e. y_{0} = y_{N+1} = 0).
    """
    return (1+mu)*diag_ones(N) - mu * offd_ones(N)


def Vij_periodic(N, mu=0.2):
    """
    Potential matrix with periodic boundary conditions (i.e. spring connects y_1 to y_N).
    """
    return Vij_closed(N,x=x)  - mu * corner_ones(N)


def Vij_open(N, mu=0.2):
    """
    Potential matrix with open boundary conditions (i.e. y_1 and y_N only connected on one side).
    """
    return Vij_closed(N,x=x)  - mu * firstlast_ones(N)


def V(N=2, mu=.2, bcs="closed"):
    if bcs=="closed":
        return Vij_closed(N, mu)
    if bcs=="open":
        return Vij_open(N, mu)
    if bcs=="periodic":
        return Vij_periodic(N, mu)


#####################################################################3
