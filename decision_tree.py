import numpy as np 
import matplotlib.pyplot as plt

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.depth = 0
        self.type = None
        self.pos_num = 0
        self.neg_num = 0 

#helper function to decide index of best attribute 
"""
Input: data 
Output: index of attribute that is best for splitting 
"""  
def best_attribute(data):
    highest_mi = -np.inf  
    idx_attr = -1 
    for i in range( data.shape[1] - 1 ):
        mi = cal_mi(data[:,i], data[:,-1])
        if mi > highest_mi:
            highest_mi  = mi
            idx_attr = i
    if highest_mi == 0 :
        idx_attr = None 
    return idx_attr

#helper function to calculate entropy of data 
def cal_entropy(label_column):
    count = 0 
    neg_sum = 0 
    pos_sum = 0 
    entropy = 0
    if len(label_column) == 0 :
        return 0 
    else:
        for i in range(len(label_column)):
            count += 1 
            if label_column[i] == 0 :
                neg_sum += 1
            if label_column[i] == 1 :
                pos_sum += 1
        pos_fraction = pos_sum / count 
        neg_fraction = neg_sum / count 
        if pos_fraction == 0:
            entropy = - neg_fraction * np.log2(neg_fraction)
        elif neg_fraction == 0:
            entropy = -pos_fraction * np.log2(pos_fraction)
        else:
            entropy = -pos_fraction * np.log2(pos_fraction) - neg_fraction * np.log2(neg_fraction)
        return entropy 

def cal_mi(data_column, label_column):
    label_lc = label_column[(data_column == 0 )]
    label_rc = label_column[(data_column == 1 )]
    parent_entropy = cal_entropy(label_column)
    lc_entropy = cal_entropy(label_lc)
    rc_entropy = cal_entropy(label_rc)
    lc_fraction = len(label_lc) / (len(label_lc) + len(label_rc))
    rc_fraction = 1 - lc_fraction 
    weighted_entropy = lc_fraction * lc_entropy + rc_fraction * rc_entropy
    mi = parent_entropy - (weighted_entropy)
    return mi 

def majority_vote(data):
    if len(data[(data[: , -1]) == 1]) >= len(data[(data[: , -1]) == 0]):
        vote = 1 
    else:
        vote = 0 
    return vote 

def build_tree(data, max_depth, parent_depth = -1):
    new_node = Node()
    new_node.depth = parent_depth + 1
    #base case 
    #1. reach user's specified max_depth 
    #2. All labels are identical (entropy == 0 )
    #3. No more data to split (only 1 sample in a node)
    #4. Stop when mutual information == 0 
    if len(data) ==  1 :
        new_node.type = 'leaf'
        new_node.vote = data[0, -1] 
        return new_node
    if  max_depth == 0:
        new_node.type = 'leaf'
        new_node.vote = majority_vote(data)
        return new_node 
    entropy = cal_entropy(data[:,-1])
    if entropy == 0:
        new_node.type = 'leaf'
        new_node.vote = data[0, -1]  #since all data is the same (so just pick at zero idx)
        return new_node
    if best_attribute(data) == None:
        new_node.type = 'leaf'
        new_node.vote = majority_vote(data)
        return new_node
    else:
        new_node.type = 'internal'
        splitting_attr = best_attribute(data) 
        new_node.attr = splitting_attr
        lc_data = data[(data[:, splitting_attr] == 1)]
        rc_data = data[(data[:, splitting_attr] == 0)] 
        new_node.left = build_tree(lc_data, max_depth -1, new_node.depth)
        new_node.right = build_tree(rc_data, max_depth -1, new_node.depth) 
        return new_node

def predict(node, example):
    if (node.vote != None):
        return int(node.vote) 
    else:
        if example[node.attr] == 1: 
            node = node.left
        elif example[node.attr] == 0:
            node = node.right
        return predict(node, example)

def count_middle(data,node):
    '''
    input: node, data
    todo: - update node.pos_num & node.neg_num
          - update node.left.pos_num/neg_num and node.right.pos_num/neg_num 
    '''
    neg, pos = 0, 0 
    if node.type == 'leaf':
        label = data[:, -1]
        for i in label:
            if i ==  0:
                neg += 1 
            else:
                pos+=1 
        node.pos_num = pos
        node.neg_num = neg

    if node.type == 'internal':
        label = data[:, -1]
        for i in label:
            if i ==  0:
                neg += 1 
            else:
                pos+=1 
        node.pos_num = pos
        node.neg_num = neg
        lc_data = data[(data[:, node.attr] == 1)]
        rc_data = data[(data[:, node.attr] == 0)] 
        count_middle(lc_data, node.left)
        count_middle(rc_data, node.right)

def dfs_print(node, print_out):
    dash = "| "
    print_out += f'[{node.neg_num} 0/{node.pos_num} 1]\n'
    if node.type == 'leaf':
        return print_out
    if node.type == 'internal':
        print(node.attr)
        print_out += f'{dash*(node.depth+1)}{header[node.attr]} = 0: '
        print_out = dfs_print(node.right, print_out)
        print_out += f'{dash*(node.depth+1)}{header[node.attr]} = 1: '
        print_out = dfs_print(node.left, print_out)
        return print_out

def cal_error(truth, pred):
    error = 0 
    count = 0 
    for i in range(len(truth)):
        count += 1 
        if truth[i] != pred[i]:
            error += 1
    return error/count

def predict_output(data, node):
    y_hat_lst = [] 
    for i in range(len(data)): 
        y_hat = predict(node, data[i])
        y_hat_lst.append(y_hat)
    return y_hat_lst

def plot_errors(train_data, test_data):
    max_depth = train_data.shape[1] - 1
    depth = range(max_depth)
    train_error = []
    test_error = [] 
    for each_depth in range(len(depth)):
        trained_tree = build_tree(train_data, each_depth)
        train_y_hat_lst = predict_output(train_data, trained_tree)
        test_y_hat_lst =  predict_output(test_data, trained_tree)
        train_error.append(cal_error(train_data[:, -1], train_y_hat_lst))
        test_error.append(cal_error(test_data[:, -1], test_y_hat_lst ))
    plt.plot(depth,train_error, label = "training errors")
    plt.plot(depth, test_error, label = "testing_errors")
    plt.xlabel('depth')
    plt.ylabel('error rate')
    plt.title("training errors and testing errors against depth")
    plt.legend()
    plt.savefig('errors_plot.png')
    plt.show()

import sys
if __name__ == '__main__':
    args = sys.argv
    train_input = args[1]
    test_input = args[2]
    max_depth = args[3]
    train_out = args[4]
    test_out = args[5]
    metrics_out = args[6]

    #read in all args
    with open(train_input, 'r') as input:
        for line in input:
            header = line.split('\t')
            break 
    print(header)

    train_input = np.genfromtxt(train_input, delimiter= '\t', skip_header = 1)
    test_input = np.genfromtxt(test_input, delimiter= '\t', skip_header = 1)
  
    max_depth = int(max_depth)

    trained_tree = build_tree(train_input, max_depth)

    train_y_hat_lst = predict_output(train_input, trained_tree)
    test_y_hat_lst = predict_output(test_input, trained_tree)

    with open(train_out, 'w') as train_out:
        for i in range(len(train_y_hat_lst)):
            train_out.write(str(train_y_hat_lst[i])+'\n') 

    with open(test_out, 'w') as test_out:
        for i in range(len(test_y_hat_lst)):
            test_out.write(str(test_y_hat_lst[i])+'\n') 
            
    with open(metrics_out, 'w') as output:
        train_error = cal_error(train_input[:, -1], train_y_hat_lst)
        test_error = cal_error(test_input[:, -1], test_y_hat_lst ) 
        sent = "error(train): " + str(train_error) + "\n" +  "error(test): " + str(test_error)
        output.write(sent) 

    count_middle(train_input, trained_tree)
    print_out = dfs_print(trained_tree, "")
    print(print_out)
    plot_errors(train_input, test_input)

    
