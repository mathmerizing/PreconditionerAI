from scipy.sparse import dok_matrix , lil_matrix, csc_matrix , identity , linalg, save_npz, load_npz
import os
import numpy as np
from Dataset.generateRandSparse import CustomSparse
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm

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
    custom = CustomSparse(1000,0.01)
    custom.create([1,99])

    p = Preconditioner(custom.A)
    p.print()

    p.invertDiagonal()
    print(p * custom.A)

    custom.cuthill()

    inp = np.array(custom.A.todense())
    inp2 = inp != 0.0

    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(inp2)
    edges2 = feature.canny(inp2, sigma=3)

    test_img = edges2

    # display results
    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 3),
                                    sharex=True, sharey=True)

    ax1.imshow(inp2, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title(r'Canny filter, $\sigma=3$', fontsize=20)

    # hough line detection
    h, theta, d = hough_line(test_img, theta=np.linspace(-np.pi / 2, np.pi / 2, 360))
    ax4.imshow(test_img, cmap=cm.gray)
    origin = np.array((0, inp2.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax4.plot(origin, (y0, y1), '-r')
    ax4.set_xlim(origin)
    ax4.set_ylim((inp2.shape[0], 0))
    ax4.set_axis_off()
    ax4.set_title('Huch!')

    fig.tight_layout()

    plt.show()
