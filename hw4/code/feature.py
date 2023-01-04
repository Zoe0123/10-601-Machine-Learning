import csv
import numpy as np
import sys

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

def word2vec(data, feat_dict):
    feat_vecs = []
    for row in data:
        y, x = row[0], row[1]
        words = x.split()
        feat_vec = [0] * VECTOR_LEN 
        trim_count = 0
        for w in words:
            if w in feat_dict:
                feat_vec += feat_dict[w] 
                trim_count += 1
        feat_vec = np.insert(feat_vec/trim_count, 0, y)
        feat_vecs.append(np.around(feat_vec, 6))
    return feat_vecs

if __name__ == "__main__": 
    train_input = sys.argv[1]
    val_input = sys.argv[2]
    test_input = sys.argv[3]
    feat_dict_input = sys.argv[4]
    train_out = sys.argv[5]
    val_out = sys.argv[6]
    test_out = sys.argv[7]

    train_data = load_tsv_dataset(train_input)
    val_data = load_tsv_dataset(val_input)
    test_data = load_tsv_dataset(test_input)
    feat_dict = load_feature_dictionary(feat_dict_input)

    train_vecs = word2vec(train_data, feat_dict)
    file = open(train_out,"w")
    for vec in train_vecs:
        for v in vec:
            file.write("%.6f" %v + "\t")
        file.write(f"\n")
    file.close()

    val_vecs = word2vec(val_data, feat_dict)
    file = open(val_out,"w")
    for vec in val_vecs:
        for v in vec:
            file.write("%.6f" %v + "\t")
        file.write(f"\n")
    file.close()

    test_vecs = word2vec(test_data, feat_dict)
    file = open(test_out,"w")
    for vec in test_vecs:
        for v in vec:
            file.write("%.6f" %v + "\t")
        file.write(f"\n")
    file.close()


