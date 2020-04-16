# PreconditionerAI
Tried a similar approach as in:

Götz, Markus & Anzt, Hartwig. (2018). Machine Learning-Aided Numerical Linear Algebra: Convolutional Neural Networks for the Efficient Preconditioner Generation. 10.13140/RG.2.2.30244.53122. 

Started off by handcrafting a block detection algorithm to be used for a Block Jacobi and an overlapping additive Schwarz smoother. 

Enhancements: use supervariable blocking ?!

Tested out approach on washing line matrix, a synthesized matrix which has been created from a diagonal matrix through Jacobi rotations and on a matrix from a FEM discretization of the Poisson problem.
Our approach showed small improvements in the condition number reduction compared to a Jacobi preconditioner, but was still not sufficiently good enough to be used in practice.
