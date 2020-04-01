import numpy as np
import random
import math
from random import randint
from scipy.sparse import dok_matrix , csc_matrix , identity , linalg, save_npz, load_npz
from scipy.sparse.csgraph import reverse_cuthill_mckee
from my_utils import timeit
import logging
import os
import matplotlib.pyplot as plt

# reference: https://en.wikipedia.org/wiki/Jacobi_rotation
def jacobiRotation(dim):
    l = randint(0,dim-1)
    k = randint(0,dim-1)
    Q = identity(dim,dtype=np.float32,format = "dok")
    theta = random.uniform(0,2*math.pi)
    c = math.cos(theta)
    s = math.sin(theta)

    Q[k,k] = c
    Q[l,l] = c
    Q[k,l] = s
    Q[l,k] = -s

    return Q.tocsc()

# class for artificially created matrices
# start with diagonal matrix with entries/ eigenvalues in certain range or list
# then apply a few Jacobi Rotations until matrix is not too sparse anymore
class CustomSparse:
    def __init__(self,dim=1,prob=0.0):
        self.dim = dim
        self.prob = prob
        self.A = csc_matrix((dim,dim),dtype=np.float32)
        self.Q = identity(dim,dtype=np.float32,format = "csc")
        self.D = identity(dim,dtype=np.float32,format = "csc")
        self.perm = identity(dim,dtype=np.float32,format = "csc")

    @timeit
    def create(self,eigenvalues,isList = False):
        diag = np.zeros(self.dim,dtype=np.float32)
        if isList:
            for i in range(self.dim):
                diag[i] = eigenvalues[i]
        else:
            a,b = eigenvalues
            for i in range(self.dim):
                diag[i] = random.uniform(a,b)

        self.D.setdiag(diag)
        self.A = self.D.copy()

        # while matrix is too sparse, apply jacobi rotations
        while self.A.count_nonzero()/(float)(self.dim*self.dim) <= self.prob :
            tempQ = jacobiRotation(self.dim)
            self.A = tempQ.transpose() * self.A * tempQ
            self.Q = self.Q * tempQ

        self.A = self.Q.transpose() * self.D * self.Q

    def inverse_D(self):
        diagonal = self.D.diagonal()
        tempD = csc_matrix(self.D)
        for i in range(self.dim):
            tempD[i,i] = 1/diagonal[i]
        return tempD

    @timeit
    def invert(self):
        return self.perm.transpose() * self.Q.transpose() * self.inverse_D() * self.Q *self.perm 

    def cuthill(self):
        new_order = reverse_cuthill_mckee(self.A)
        self.perm.indices = new_order.take(self.perm.indices)
        self.perm = self.perm.tocsc()

        self.A = self.perm.transpose()*self.A*self.perm
        """
        self.A.indices = new_order.take(self.A.indices)
        self.A = self.A.tocsc()
        self.A.indices = new_order.take(self.A.indices)
        self.A = self.A.tocsc()
        """
        #self.Q = self.perm*self.Q

        """
        self.Q.indices = new_order.take(self.Q.indices)
        self.Q = self.Q.tocsc()
        self.Q.indices = new_order.take(self.Q.indices)

        self.D.indices = new_order.take(self.D.indices)
        self.D = self.D.tocsc()
        self.D.indices = new_order.take(self.D.indices)
        """


    def save(self,foldername):
        try:
            if not os.path.exists(foldername):
                os.mkdir(foldername)
        except OSError:
            logging.error(f"Creation of the directory {foldername} failed!")
        A_path = os.path.join(foldername,"A.npz")
        Q_path = os.path.join(foldername,"Q.npz")
        D_path = os.path.join(foldername,"D.npz")

        save_npz(A_path, self.A)
        save_npz(Q_path, self.Q)
        save_npz(D_path, self.D)

    def load(self,foldername):
        A_path = os.path.join(foldername,"A.npz")
        Q_path = os.path.join(foldername,"Q.npz")
        D_path = os.path.join(foldername,"D.npz")

        self.A = load_npz(A_path)
        self.Q = load_npz(Q_path)
        self.D = load_npz(D_path)

        self.dim = self.A.shape[0]
        # self.prob can't be reconstructed and doesn't need to be

    def small_matrices(self):
        # yield 128 x 128 matrices

        # if dim is not a multiple of 128, the last matrix is
        # "padded by an identity matrix"
        non_zero_indices = self.A.nonzero()
        i = 0
        if self.dim % 128 != 0:
            logging.debug("self.dim not divisible by 128.")
            logging.debug("Last matrix will be padded.")

        for k in range(math.ceil(self.dim / 128.0)):
            logging.debug(f"Big matrix indices: [{k*128},{k*128+127}]")
            tmp = np.identity(128)

            upper_limit = min((k+1)*128,self.dim)
            start = k*128

            while(i < len(non_zero_indices[0]) and non_zero_indices[0][i] < upper_limit and non_zero_indices[0][i] >= start):
                if (non_zero_indices[1][i] >= start and non_zero_indices[1][i] < upper_limit ):
                    # write in new matrix
                    tmp[non_zero_indices[0][i]-start,non_zero_indices[1][i]-start] = self.A[non_zero_indices[0][i],non_zero_indices[1][i]]
                    i += 1
                else:
                    i += 1

            yield tmp


    def preconditioned_cond(self,precond,precond_inv):
        condition_num  = linalg.norm(precond_inv * self.A)
        condition_num *= linalg.norm(self.invert() * precond)
        return condition_num

    def condition(self):
        maxi = max(abs(self.D.diagonal()))
        mini = min(abs(self.D.diagonal()))
        return maxi / mini

    def print(self):
        logging.debug(f"A = \n {self.A.todense()}")




if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR , format='[%(asctime)s] - [%(levelname)s] - %(message)s')

    # creating a test object
    custom = CustomSparse(25,0.3)
    custom.create([1,99])
    logging.error(f"Rounding error while inverting: {(custom.invert().dot(custom.A)-identity(custom.dim)).max()}")
    # save a CustomSparse object
    # custom.save("test1")

    # test loading a CustomSparse object
    # new_custom = CustomSparse()
    # new_custom.load("test1")
    # logging.debug(f"Loading error: {(custom.invert()-new_custom.invert()).max()}") --> dimension mismatch, since new_custom.perm not loaded properly!!!
    """
    for matrix1,matrix2 in zip(custom.small_matrices(),custom.small_matrices2()):
        print((matrix1-matrix2).max())
        print("")
    """
    for matrix in custom.small_matrices():
        print("")


    # custom.print()
    """
    fig, axs = plt.subplots(2, 2)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    x = np.random.randn(20, 20)
    x[5, :] = 0.
    x[:, 12] = 0.


    ax1.spy(custom.A.todense(), precision=0.1)
    ax2.spy(custom.perm.todense(), precision=0.1, markersize=5)
    """
    M = custom.A.copy()
    print(custom.D.diagonal())
    print(custom.A.todense())
    custom.cuthill()
    print(custom.A.todense())
    """
    ax3.spy(custom.A.todense(), precision=0.1)
    ax4.spy(custom.perm.todense(), precision=0.1, markersize=5)



    plt.show()
    """

    logging.error(f"Rounding error while inverting: {(custom.invert().dot(custom.A)-identity(custom.dim)).max()}")

    fig, axs = plt.subplots(3)
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax1.spy(custom.perm*M*custom.perm.transpose(), precision = 0.001)    #(M- custom.Q.transpose() * custom.D * custom.Q).todense(), precision=0.01) #
    ax2.spy(custom.A.todense(), precision=0.001)
    ax3.spy(custom.perm*M*custom.perm.transpose()-custom.A.todense(), precision=0.001)
    plt.show()
