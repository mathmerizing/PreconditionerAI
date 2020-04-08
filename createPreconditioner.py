from scipy.sparse import dok_matrix , lil_matrix, csc_matrix , identity , linalg, save_npz, load_npz
import os
import numpy as np
from Dataset.generateRandSparse import CustomSparse
from Dataset.my_utils import millis, timeit
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm
import os
import logging

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
    logging.basicConfig(level=logging.DEBUG , format='[%(asctime)s] - [%(levelname)s] - %(message)s')

    # creating a test object
    custom = CustomSparse(1000,0.2)
    custom.create([1,99])

    """
    p = Preconditioner(custom.A)
    #p.print()
    p.invertDiagonal()
    #print(p * custom.A)
    """

    # reduce bandwidth
    custom.cuthill()

    # find Preconditioner for sparse_matrix
    sparse_matrix = load_npz(os.path.join("DealScripts","dataset", "cuthill", "system_1313_0.300000_0.200000.npz")) # custom.A

    block_coordinates = []
    cur = 0
    # pic_size is only for visualization
    pic_size = 32

    logging.debug("Starting sophisticated block search")

    # find blocks using custom simplified edge detection (with L-shapes)
    start_time = millis()
    while (cur < sparse_matrix.shape[0]):
        #sparsity_pattern = sparse_matrix[cur:cur+pic_size,cur:cur+pic_size].todense()!=0.0

        # average over L shape with i elements to the right and (i-1) elements to the bottom
        values = []
        for i in range(1,min(64,sparse_matrix.shape[0]-cur)):
            L_to_right = abs(sparse_matrix[cur:cur+i,cur]!=0.0).mean()*i
            L_to_bottom = abs(sparse_matrix[cur+i-1,cur+1:cur+i]!=0.0).mean()*(i-1) if i > 1 else 0
            values.append((L_to_right+L_to_bottom) / (2*i-1))

        # calculate differences between consecutive values
        diffs = np.array(values[1:]) - np.array(values[:-1])

        # multiply diffs by some scaling function
        scaled_diffs = []
        for i, num in enumerate(diffs):
            scaled_diffs.append(num*min(i+1,10))
        scaled_diffs = np.array(scaled_diffs)

        # find the index of the biggest scaled_diffs value
        if (abs(sparse_matrix[cur+1:cur+20,cur]!=0.0).sum()+abs(sparse_matrix[cur,cur+1:cur+20]!=0.0).sum() == 0):
            big_jump = 0
        else:
            big_jump = 0 if len(scaled_diffs) == 0 else np.argmax(abs(scaled_diffs))

        # save the corners of the block
        block_coordinates.append((cur,cur+big_jump))
        cur += big_jump + 1;

        """
        # visualize sparsity_pattern and block_coordinates predictions
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=False, sharey=False)
        ax1.plot(list(range(pic_size)), abs(diffs[:pic_size]))
        ax1.set_xticks(np.arange(pic_size))

        ax2.plot(list(range(pic_size)), abs(scaled_diffs[:pic_size]))
        ax2.set_xticks(np.arange(pic_size))

        ax3.imshow(sparsity_pattern, cmap=plt.cm.gray, extent=[0, pic_size-1, 0, pic_size-1])
        ax3.set_xticks(np.arange(pic_size))
        ax3.set_yticks(np.arange(pic_size))

        fig.suptitle(big_jump)
        plt.show()
        """
    logging.debug("Starting Preconditioner")
    prec_test = Preconditioner(sparse_matrix)
    prec_test.invert(block_coordinates)
    print("Created Preconditioner")
    logging.debug(f"Block Preconditioner took {(millis()-start_time)/1000} seconds.")

    logging.debug(f"Condition sparse: {np.linalg.cond((sparse_matrix).todense())}")
    logging.debug(f"Condition block Jacobi: {np.linalg.cond((prec_test * sparse_matrix).todense())}")

    prec_bench = Preconditioner(sparse_matrix)
    prec_bench.invertDiagonal()
    logging.debug(f"Condition Jacobi: {np.linalg.cond((prec_bench * sparse_matrix).todense())}")

    inv_start = millis()
    np.linalg.inv(sparse_matrix.todense())
    logging.debug(f"invert took {(millis()-inv_start)/1000} seconds.")
