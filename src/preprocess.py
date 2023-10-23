#! /bin/python3

def data_to_csv(filename, dim):
    filepath = 'data/raw/' + filename
    out_filepath = 'data/processed/' + filename + '.csv'
    in_fp = open(filepath, 'r')
    out_fp = open(out_filepath, 'w+')
    for line in in_fp:
        bits = line.strip().split(' ')
        i = 1
        k = 1
        full_bits = [bits[0]]
        while i < dim and k < len(bits):
            n, b = bits[k].split(':')
            if int(n) == i:
                full_bits.append(b)
                k += 1
            else:
                full_bits.append('0')
            i += 1
        while len(full_bits) < dim:
            full_bits.append('0')
        full_bits.append(full_bits.pop(0))
        newline = ','.join(full_bits)
        out_fp.write(newline + '\n')
    in_fp.close()
    out_fp.close()

def npy_csr_to_csv(filename):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from scipy.sparse import csr_matrix
    import pandas as p
    filepath = 'data/raw/' + filename
    out_filepath = 'data/processed/' + filename + '.csv'

    npz = np.load(filepath)
    sparse_matrix = csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
    dense_matrix = sparse_matrix.todense()
    np.savetxt(out_filepath, dense_matrix, delimiter=',')

def npy_csr_to_coo_csv(filename):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io as sio
    from scipy.sparse import csr_matrix
    import itertools
    filepath = 'data/raw/' + filename
    out_filepath = 'data/processed/' + filename + 'coo.csv'

    npz = np.load(filepath)
    sparse_matrix = csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
    coo_matrix = sparse_matrix.tocoo()
    print("Finished COO conversion")

    row_map = {}
    for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        row_map[i] = row_map.get(i, []) + [(j, v)]

    with open(out_filepath, 'w+') as fp:
        for i in range(coo_matrix.shape[0]):
            if i in row_map:
                row = row_map[i]
                row = [f"{entry[0]}:{entry[1]}" for entry in row]
                fp.write(','.join(row) + '\n')
            else:
                fp.write('\n')


if __name__ == '__main__':
    # print('=> Preprocessing data...')
    # print('. processing usps')
    # data_to_csv('usps', 256)
    # print('. processing mnist')
    # data_to_csv('mnist', 784)
    # print('=> Preprocessing completed.')

    # Takes long time to process, uncomment if really needed!
    # print('. processing mnist8m')
    # data_to_csv('mnist8m', 784)
    # print('=> Preprocessing completed.')
    print('. processing Data_C5')
    npy_csr_to_csv('Data_C5.npz')
    print('=> Preprocessing completed.')
    print('. processing Data_C6')
    npy_csr_to_csv('Data_C6.npz')
    print('=> Preprocessing completed.')