import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_indices, tags_to_indices, hmmprior, hmmemit, hmmtrans = get_inputs()

    # Initialize the initial, emission, and transition matrices
    n_tag = len(tags_to_indices)
    n_word = len(words_to_indices)
    p = np.zeros((n_tag, ))
    A = np.zeros((n_tag, n_word))
    B = np.zeros((n_tag, n_tag))

    # Increment the matrices
    N = len(train_data)
    for i in range(N):
        init = train_data[i][0]
        t_ind = tags_to_indices[init[1]]
        p[t_ind] += 1
    for i in range(N):
        sentence = train_data[i]
        M = len(sentence)
        for j in range(M-1):
            t1, t2 = sentence[j][1], sentence[j+1][1]
            t1_ind, t2_ind = tags_to_indices[t1], tags_to_indices[t2]
            B[t1_ind, t2_ind] += 1
            w1 = sentence[j][0]
            w1_ind = words_to_indices[w1]
            A[t1_ind, w1_ind] += 1
        t, w = sentence[M-1][1], sentence[M-1][0]
        t_ind, w_ind = tags_to_indices[t], words_to_indices[w]
        A[t_ind, w_ind] += 1
        

    # Add a pseudocount
    p += 1
    A += 1
    B += 1

    # Save your matrices to the output files --- the reference solution uses 
    p = p / np.sum(p)
    A = A / np.sum(A, axis=1)[:, np.newaxis]
    B = B / np.sum(B,  axis=1)[:, np.newaxis]

    np.savetxt(hmmprior, p)
    np.savetxt(hmmemit, A)
    np.savetxt(hmmtrans, B)

    # np.savetxt (specify delimiter="\t" for the matrices)
    
    # pass

    
    # n_tag = len(tags_to_indices)
    # n_word = len(words_to_indices)
    # p = np.zeros((n_tag, ))
    # A = np.zeros((n_tag, n_word))
    # B = np.zeros((n_tag, n_tag))

    # # Increment the matrices
    # N_list = [10, 100, 1000, 10000]
    # for N in N_list:
    #     for i in range(N):
    #         init = train_data[i][0]
    #         t_ind = tags_to_indices[init[1]]
    #         p[t_ind] += 1
    #     for i in range(N):
    #         sentence = train_data[i]
    #         M = len(sentence)
    #         for j in range(M-1):
    #             t1, t2 = sentence[j][1], sentence[j+1][1]
    #             t1_ind, t2_ind = tags_to_indices[t1], tags_to_indices[t2]
    #             B[t1_ind, t2_ind] += 1
    #             w1 = sentence[j][0]
    #             w1_ind = words_to_indices[w1]
    #             A[t1_ind, w1_ind] += 1
    #         t, w = sentence[M-1][1], sentence[M-1][0]
    #         t_ind, w_ind = tags_to_indices[t], words_to_indices[w]
    #         A[t_ind, w_ind] += 1
        

    #     # Add a pseudocount
    #     p += 1
    #     A += 1
    #     B += 1

    #     # Save your matrices to the output files --- the reference solution uses 
    #     p = p / np.sum(p)
    #     A = A / np.sum(A, axis=1)[:, np.newaxis]
    #     B = B / np.sum(B,  axis=1)[:, np.newaxis]

    #     np.savetxt(f"en_data/hmminit_{N}.txt", p)
    #     np.savetxt(f"en_data/hmmemit_{N}.txt", A)
    #     np.savetxt(f"en_data/hmmtrans_{N}.txt", B)