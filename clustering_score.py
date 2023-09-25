import sys
import numpy as np
from s_dbw import S_Dbw, SD
from scipy.sparse import csr_matrix
import time


if __name__ == '__main__':
    npz = np.load(sys.argv[1])
    print(f"Loading data from {sys.argv[1]}")
    start = time.time()
    data = csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
    data = data.todense()
    end = time.time()
    print(f"Finished loading data in {end - start} seconds")

    labels = np.genfromtxt(sys.argv[2], delimiter=',')

    print(data.shape)
    print(labels.shape)

    print("Calculating S_Dbw score")
    start = time.time()
    score = S_Dbw(data, labels, nearest_centr=False, method='Halkidi', alg_noise='bind', centr='mean')
    # score = SD(data, labels)
    end = time.time()
    print(f"Finished calculating S_Dbw score in {end - start} seconds")


