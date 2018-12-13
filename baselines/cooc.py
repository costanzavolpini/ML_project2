#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
from astropy.wcs.docstrings import row


def main():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print("vocab size: ", vocab_size)

    data, row, col = [], [], []
    counter = 1

    #for fn in ['train_pos_small.txt', 'train_neg_small.txt']:
    for fn in ['train_pos_full.txt', 'train_neg_full.txt']:
        with open(fn) as f:
            for line in f:
                #print("line: ", line)
                tokens = [vocab.get(t, -1) for t in line.strip().split()] # index in vocab of a certain element in the line. Smaller index = higher occurrence
                tokens = [t for t in tokens if t >= 0]
                #print("tokens: ", tokens)
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
                
    print("data len: ", len(data))
    print("row: ", len(row))
    print("col: ", len(col))
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    
#     print("vocab: ", vocab)
    
    with open('cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
