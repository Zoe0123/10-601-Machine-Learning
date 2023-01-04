import sys
import numpy as np
import statistics as st
# import matplotlib.pyplot as plt

class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """
    def __init__(self, att, v):
        self.left = None
        self.right = None
        self.attr = att
        self.vote = v

def load_data(fpath):
    f = open(fpath)
    x_names = f.readline().strip('\n').split('\t')[:-1]
    f.close()  
    data = np.loadtxt(fpath,dtype=str,delimiter='\t',skiprows=1)
    x, y = data[:, :-1], data[:, -1]
    return np.array(x_names), np.array(x, dtype=int), np.array(y, dtype=int)

def train(x_names, train_x, train_y, depth):
    """
    Assume depth >= 0

    Return: a Decision Tree with max-depth <depth>.
    """
    tree = Node(None, None)
    # majority vote
    if depth == 0 or train_x.size == 0: 
        count1 = np.count_nonzero(train_y)
        if count1 >= len(train_y) - count1:
            tree.vote = 1
        else:
            tree.vote = 0
    else:
        Is = []
        # mutual information for each x and Y
        for i in range(train_x.shape[1]):
            Is.append(mutual_info(train_y, train_x[:, i]))

        # set node attribute as the name of x splitted on (the x gives the max info)
        ind_split = np.argmax(np.array(Is))
        tree.attr = x_names[ind_split]
        if type(ind_split) != np.int64:
            ind_split = min(ind_split)

        # stop split if max I <= 0
        if Is[ind_split] <= 0:
            count1 = np.count_nonzero(train_y)
            if count1 >= len(train_y) - count1:
                tree.vote = 1
            else:
                tree.vote = 0
            return tree 

        tree.left, tree.right = Node(None, None), Node(None, None)
        # find index of x = 1 and x = 0
        ind_x1 = np.argwhere(train_x[:, ind_split] == 1)
        ind_x0 = np.argwhere(train_x[:, ind_split] == 0)
        
        # recursion
        tree.left = train(np.delete(x_names, ind_split, 0), np.delete(np.delete(train_x, ind_x0, 0), ind_split, 1), 
        np.delete(train_y, ind_x0, 0), depth-1)
          
        tree.right = train(np.delete(x_names, ind_split, 0), np.delete(np.delete(train_x, ind_x1, 0), ind_split, 1), 
        np.delete(train_y, ind_x1, 0), depth-1)
    return tree

def Entropy(Y):
    n = Y.shape[0]
    prob_1 = np.count_nonzero(Y)/n
    prob_0 = 1 - prob_1
    H_y0, H_y1 = 0, 0
    if prob_0 != 0:
        H_y0 = - prob_0 * np.log2(prob_0)
    if prob_1 != 0:
        H_y1 = - prob_1 * np.log2(prob_1)
    return H_y0 + H_y1

def mutual_info(Y, x):
    n = x.shape[0]
    xandY = x + Y
    
    # 1. calculate H_Y
    H_y = Entropy(Y)
     
    # 2. calculate H_yx
    H_yx1, H_yx0 = 0, 0
    count_x1 = np.count_nonzero(x)
    if count_x1 != 0:
        # P(Y=1|x=1) and P(Y=0|x=1), then calculate H(Y|X=x1)
        count_y1x1 = np.count_nonzero(xandY == np.array([2]*n))
        count_y0x1 = count_x1 - count_y1x1
        prob_y1x1, prob_y0x1 = count_y1x1/count_x1, count_y0x1/count_x1

        H_y1x1, H_y0x1 = 0, 0
        if prob_y1x1 != 0:
            H_y1x1 = - prob_y1x1 * np.log2(prob_y1x1)
        if prob_y0x1 != 0:
            H_y0x1 = - prob_y0x1 * np.log2(prob_y0x1)
        H_yx1 = H_y1x1 + H_y0x1 

    count_x0 = n - count_x1
    if count_x0 != 0:
        # P(Y=1|x=1) and P(Y=0|x=1), then calculate H(Y|X=x1)
        count_y0x0 = np.count_nonzero(xandY == np.array([0]*n))
        count_y1x0 = count_x0 -  count_y0x0 
        prob_y1x0, prob_y0x0 = count_y1x0/count_x0, count_y0x0/count_x0

        H_y1x0, H_y0x0 = 0, 0
        if prob_y1x0 != 0:
            H_y1x0 = - prob_y1x0 * np.log2(prob_y1x0)
        if prob_y0x0 != 0:
            H_y0x0 = - prob_y0x0 * np.log2(prob_y0x0)
        H_yx0 = H_y1x0 + H_y0x0 
 
    H_yx = count_x1/n * H_yx1 + count_x0/n * H_yx0

    return H_y - H_yx

def predict(tree, x, x_names):
    # example is a dictionary which holds the attributes and the
    # values of the attribute (ex. example[’X’] = 0)
    preds = []
    for row in x:
        preds.append(_predict(tree, row, x_names))
    return preds

def _predict(tree, row, x_names):
    if tree.vote is not None:
        return tree.vote

    ind_split = np.argwhere(x_names==tree.attr)
    if row[ind_split] == 1:
        return _predict(tree.left, row, x_names) 
    else:
        return _predict(tree.right, row, x_names) 
        

def evaluate(pred, y):
    n = y.shape[0]
    return (n-np.count_nonzero(y==pred))/n

def print_tree(tree, x_names, x, y, depth):
    n = y.shape[0]
    y_count1 = np.count_nonzero(y)
    y_count0 = n - y_count1
    print(f"[{y_count0} 0/{y_count1} 1]")
    if tree.attr is not None:
        _print_tree(tree, x_names, x, y, depth)
    
def _print_tree(tree, x_names, x, y, depth):    
    n = y.shape[0]
    
    ind_split = np.argwhere(x_names==tree.attr)[0][0]
    x_split = x[:, ind_split]

    count_x1 = np.count_nonzero(x_split)
    count_y1x1 = np.count_nonzero(y[np.argwhere(x_split==1)]==1)
    count_y0x1 = count_x1 - count_y1x1
    
    ind_x1 = np.argwhere(x_split == 1)
    ind_x0 = np.argwhere(x_split == 0)
    print("| " * depth + f"{tree.attr} = 1: [{count_y0x1} 0/{count_y1x1} 1]")
    if tree.left is not None and tree.left.attr is not None:
        _print_tree(tree.left, np.delete(x_names, ind_split, 0), np.delete(np.delete(x, ind_x0, 0), ind_split, 1), 
        np.delete(y, ind_x0, 0), depth+1)

    count_x0 = n - count_x1
    count_y0x0 = np.count_nonzero(y[np.argwhere(x_split==0)]==0)
    count_y1x0 = count_x0 -  count_y0x0 
    print("| " * depth + f"{tree.attr} = 0: [{count_y0x0} 0/{count_y1x0} 1]")
    if tree.right is not None and tree.right.attr is not None:
        if tree.attr in x_names:
            x_names = np.delete(x_names, ind_split, 0)
            x = np.delete(x, ind_split, 1)
        _print_tree(tree.right, x_names, np.delete(x, ind_x1, 0), np.delete(y, ind_x1, 0), depth+1)

if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metric_out = sys.argv[6]

    x_names, test_x, test_y = load_data(test_input)
    x_names, train_x, train_y = load_data(train_input)

    tree = train(x_names, train_x, train_y, int(max_depth))

    pred_train = predict(tree, train_x, x_names)
    file = open(train_out,"w")
    for i in pred_train:
        file.writelines([str(i)+'\n'])
    file.close()

    pred_test = predict(tree, test_x, x_names)
    file = open(test_out,"w")
    for i in pred_test:
        file.writelines([str(i)+'\n'])
    file.close()

    err_train = evaluate(pred_train, train_y)
    err_test = evaluate(pred_test, test_y)
    file = open(metric_out,"w")
    lines = [f'error(train): {err_train}\n',f'error(test): {err_test}\n']
    file.writelines(lines)
    file.close()

    print_tree(tree, x_names, train_x, train_y, 1)
    

    # # plot all possible max_depths with train and test errors
    # errs_train, errs_test = [], []
    # for d in range(len(x_names)+1):
    #     tree = train(x_names, train_x, train_y, d)
    #     pred_train = predict(tree, train_x, x_names)
    #     pred_test = predict(tree, test_x, x_names)

    #     errs_train.append(evaluate(pred_train, train_y))
    #     errs_test.append(evaluate(pred_test, test_y))

    # plt.plot(np.arange(len(x_names)+1), errs_train, '-bo')
    # plt.plot(np.arange(len(x_names)+1), errs_test, '-go')
    # plt.xticks(np.arange(0, len(x_names)+1, 1))
    # plt.xlabel('max_depth')
    # plt.ylabel('error')
    # plt.title('train and test errors for each max_depth')
    # plt.legend(['train error', 'test error'])
    # plt.savefig('heart.png')