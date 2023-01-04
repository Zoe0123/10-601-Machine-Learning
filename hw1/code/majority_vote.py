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

if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metric_out = sys.argv[5]

    train_x, train_y = load_data(train_input)
    test_x, test_y = load_data(test_input)

    mode_y = train(train_y)
    file = open(train_out,"w")
    file.writelines([str(mode_y)+'\n']* train_y.shape[0])
    file.close()

    pred_test = test(mode_y)
    file = open(test_out,"w")
    file.writelines([str(pred_test)+'\n']* test_y.shape[0])
    file.close()

    err_train = evaluate(mode_y, train_y)
    err_test = evaluate(pred_test, test_y)
    file = open(metric_out,"w")
    lines = [f'error(train): {err_train}\n',f'error(test): {err_test}\n']
    file.writelines(lines)
    file.close()

