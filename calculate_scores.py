import sys
import os
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
    starting_folder = sys.argv[2]

    print(f"Finished loading data in {end - start} seconds")
    for folder in os.listdir(starting_folder):
        folder_path = os.path.join(starting_folder, folder)
        if os.path.isdir(folder_path):
            scores_path = os.path.join(folder_path, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            # Iterate through all the iter.csv files and create graphs
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    score_path = os.path.join(scores_path, file)
                    labels = np.genfromtxt(file_path, delimiter=',')
                    labels = labels.astype(int)

                    score = SD(data, labels)
                    # Save score in score_path file
                    with open(score_path, 'w') as fp:
                        fp.write(str(score))

