import sys
import numpy as np
import statistics as st

def load_data(fpath):
    # df = pd.read_csv(fpath, sep='\t')
    # x, y = df.iloc[:,:-1], df.iloc[:,-1]
    data = np.loadtxt(fpath,dtype=str,delimiter='\t',skiprows=1)
    x, y = data[:, :-1], data[:, -1]
    return x, y

def train(train_y):
    modes_y = st.mode(train_y)
    return max(modes_y)

def test(mode_y):
    return mode_y

def evaluate(pred, y):
    n = y.shape[0]
    return (n-np.count_nonzero(y==str(pred)))/n

def Entropy(Y):
    n = Y.shape[0]
    ones = np.array(['1']*n)
    prob_1 = np.count_nonzero(Y==ones)/n
    prob_0 = 1 - prob_1
    H_y0, H_y1 = 0, 0
    if prob_0 != 0:
        H_y0 = - prob_0 * np.log2(prob_0)
    if prob_1 != 0:
        H_y1 = - prob_1 * np.log2(prob_1)
    return H_y0 + H_y1

if __name__ == '__main__':
    train_input = sys.argv[1]
    inspect_out = sys.argv[2]

    train_x, train_y = load_data(train_input)

    mode_y = train(train_y)
    err_train = evaluate(mode_y, train_y)
    H_y = Entropy(train_y)

    file = open(inspect_out,"w")
    lines = [f'entropy: {H_y}\n',f'error: {err_train}\n']
    file.writelines(lines)
    file.close()