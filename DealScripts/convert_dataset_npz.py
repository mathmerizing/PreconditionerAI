from scipy.sparse import dok_matrix , lil_matrix, csc_matrix , identity , linalg, save_npz, load_npz
import os
import numpy as np

dim = 1313

for root, dirs, files in os.walk("./dataset/"):
    for name in files:
        if name.endswith(".txt"):
            file_name = os.path.join(root,name)
            npz_file_name = file_name.replace(".txt", ".npz")

            matrix = lil_matrix((dim,dim),dtype=np.float32)

            with open(file_name) as f:
                for line in f:
                    try:
                        a,b = line.split(" ")
                        b = b.strip("\n")
                        a = a.strip("(").strip(")")

                        i, j = a.split(",")
                        i = int(i)
                        j = int(j)
                        val = np.float(b)

                        matrix[i,j] = val
                    except Exception as e:
                        print(e)
                        print("LINE:", line)

            save_npz(npz_file_name, matrix.tocsr())
