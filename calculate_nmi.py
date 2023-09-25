def calculate_nmi():
    # Go through all the files in the folder data/processed/cssc
    # and calculate the NMI between the ground truth in the file data/processed/small_Y.csv
    # and the clustering result in each file iter#.csv

    # The result should be a list of NMI values, one for each file
    # The list should be saved in a file called nmi.csv
    
    # Your code goes here
    import os
    import numpy as np
    import pandas as pd
    from sklearn.metrics.cluster import normalized_mutual_info_score

    # Load the ground truth
    Y = pd.read_csv("data/processed/mnist8m_Y.csv", header=None)
    Y = Y[0].values

    # Load the clustering results
    nmi = []
    for file in os.listdir("data/processed/mem_cssc_mnist8m"):
        if file.endswith(".csv"):
            # print(os.path.join("data/processed/cssc", file))
            # print(file)
            # print(file.split(".")[0])
            iter = int(file.split(".")[0][-1])
            X = pd.read_csv(os.path.join("data/processed/mem_cssc_mnist8m", file), header=None)
            X = X[0].values
            nmi.append([iter, normalized_mutual_info_score(Y, X)])
    # print(nmi)
    

    # Save the result to nmi.csv
    df = pd.DataFrame(nmi)
    df.to_csv("nmi.csv", header=None, index=None)

    # Calculate the average NMI score and deviation
    nmi = np.array(nmi)
    print("Average NMI: ", np.mean(nmi[:, 1]))
    print("NMI std: ", np.std(nmi[:, 1]))

if __name__ == "__main__":
    calculate_nmi()