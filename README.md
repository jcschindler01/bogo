# bogo
Module for calculating Bogoliubov transformation amplitudes.

To install go to the folder with setup.py and run

`$ pip install -e .`

Then you can

`>>> import bogo`

Current exports are `ab, bA, aA, eig, V`.

"Local to rotated" number state transition amplitude:

`ab(na,nb,M) = <na|nb>` 

"Sqeezing" transition amplitude:

`bA(nb,nA,M) = <nb|nA>` 

"Local to global" transition amplitude:

`aA(na,nA,M) = <na|nA>` 

In these functions `M` is the classical coupling matrix. Further:

`vals, vecs = eig(M)` gives sorted eigendecomp by increasing frequency.

`V(K,mu,bcs) = M` gives coupling matrix with
coupling strength `mu` in (0,1) and `K` total modes (`bcs` defaults to `"closed"` boundaries). 



