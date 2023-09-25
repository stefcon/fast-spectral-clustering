import sys
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

def read_positions(positions_file):
    positions = []
    with open(positions_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)
        for row in reader:
            if positions == []:
                print(row)
            positions.append(row)
    positions = np.array(positions)
    return positions


def visualize_results():
    starting_folder = sys.argv[1]
    positions_file = sys.argv[2]
    positions = np.genfromtxt(positions_file, delimiter='\t', skip_header=1)
    print(positions.shape)
    # Iterate through all the subfolders and visualize the results
    # by creating subfolder 'graphs' in each subfolder and saving
    # the graphs for each iter.csv file there.
    for folder in os.listdir(starting_folder):
        folder_path = os.path.join(starting_folder, folder)
        if os.path.isdir(folder_path):
            # Create subolder 'graphs' if it doesn't exist
            graphs_folder = os.path.join(folder_path, 'graphs')
            if not os.path.exists(graphs_folder):
                os.makedirs(graphs_folder)
            # Iterate through all the iter.csv files and create graphs
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    graph_path = os.path.join(graphs_folder, file[:-4])
                    labels = np.genfromtxt(file_path, delimiter=',')
                    try:
                        # Choose good color palette for distinguishing clusters
                        plt.scatter(positions[:, 0], positions[:, 1], c=labels, cmap=plt.cm.get_cmap('viridis', int(np.max(labels)) + 1))
                        # plt.scatter(positions[:, 0], positions[:, 1], c=labels)
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.savefig(graph_path)
                        plt.clf()
                    except Exception as e:
                        print(e)
                        print(file_path)
                        print(positions.shape)
                        print(labels.shape)
    print('Done')


if __name__ == '__main__':
    visualize_results()
