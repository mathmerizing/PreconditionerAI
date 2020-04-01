from scipy.sparse import dok_matrix , lil_matrix, csc_matrix , identity , linalg, save_npz, load_npz
import os
import numpy as np
from Dataset.generateRandSparse import CustomSparse

class Preconditioner():
    def __init__(self, system_matrix):
        self.system = system_matrix
        self.dim = self.system.shape[0]
        self.inverted = False
        self.inverse = dok_matrix((self.dim,self.dim),dtype=np.float32)

    def invert(self, block_coordinates):
        # block_coordinates = [(start_1,end_1),(start_2,end_2),...]
        for (start,end) in block_coordinates:
            system_block = self.system[start:end+1,start:end+1].A
            system_block_inv = np.linalg.inv(system_block)

            nonzero = np.nonzero(system_block_inv)
            x_coords, y_coords = nonzero[0], nonzero[1]
            for i in range(len(x_coords)):
                x,y = x_coords[i], y_coords[i]
                self.inverse[x+start,y+start] = system_block_inv[x,y]

        # convert to csc which is better for multiplication
        self.inverse = self.inverse.tocsc()
        self.inverted = True

    def invertDiagonal(self):
        # diagonal Jacobi Preconditioner
        coords = [(i,i) for i in range(self.dim)]
        self.invert(coords)

    def print(self):
        print("dim = ", self.dim)
        print("System matrix: ")
        print(self.system)
        print("Preconditioner: ")
        print(self.inverse)

    def __mul__(self, other):
        assert self.inverted, "Preconditioner hasn't been inverted."
        assert isinstance(self.inverse,csc_matrix), "Inverse is not a CSC matrix."
        print(type(self.inverse))
        return self.inverse * other


if __name__ == "__main__":
    # creating a test object
    custom = CustomSparse(100,0.1)
    custom.create([1,99])

    p = Preconditioner(custom.A)
    p.print()

    p.invertDiagonal()
    print(p * custom.A)
