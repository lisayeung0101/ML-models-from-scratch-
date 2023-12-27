import numpy as np
import argparse
import matplotlib.pyplot as plt


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta: np.ndarray,
    X : np.ndarray,     
    y : np.ndarray,  
    num_epoch : int, 
    learning_rate : float) -> None:

    for epoch in range(num_epoch):
        for i in range(len(X)):
            z =  np.dot(theta, X[i])
            gradient = (sigmoid(z) - y[i]) * X[i]
            theta = theta - learning_rate * gradient
    return theta 
            
def predict(
    theta : np.ndarray,
    X : np.ndarray) -> np.ndarray:
    y_pred = [] 
    for i in range(len(X)):
        y_val = sigmoid(np.dot(theta, X[i]))
        if y_val >= 0.5 : 
            y_hat = 1 
        else:
            y_hat = 0 
        y_pred.append(y_hat)
    return y_pred 

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray) -> float:
    err_count = 0 
    error_rate = 0 
    for i in range(len(y)):
        if y[i] != y_pred[i]:
            err_count += 1
    error_rate = err_count / len(y)
    return error_rate  


def anll(theta, X, y):
    """
    Calculates the logistic regression for a given theta and input data X, with corresponding labels y.

    Args:
    theta (array-like): the weights for the logistic regression model
    X (array-like): the input data to the logistic regression model
    y (array-like): the corresponding labels for the input data

    Returns:
    A float representing the average negative log-likelihood of the logistic regression model on the input data.
    """
    m = len(y) # number of training examples
    z = np.dot(X, theta) # calculate the linear part of the logistic regression model
    h = 1 / (1 + np.exp(-z)) # calculate the sigmoid function
    J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) # calculate the average negative log-likelihood

    return J


def anll_plot(X, y, epochs, learning_rate):
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []
    for lr in (learning_rate): 
        for epoch in range(epochs):
            z = np.dot(X, theta)
            h = sigmoid(z)
            J = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
            J_history.append(J)
            gradient = np.dot(X.T, h - y) / m
            theta = theta - learning_rate * gradient
    
        plt.plot(range(epochs), J_history , label = f' lr = {lr}')
        plt.xlabel('Number of epochs')
        plt.ylabel('Average negative log-likelihood')
        plt.show()


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=str, help='number of epochs of gradient descent to run')
    parser.add_argument("learning_rate", type=str, help='learning rate for gradient descent')
    args = parser.parse_args()

    train_input  = np.loadtxt(args.train_input, delimiter='\t', comments=None, encoding='utf-8', dtype= str)
    test_input = np.loadtxt(args.test_input, delimiter='\t', comments=None, encoding='utf-8', dtype= str)
    validation_input = np.loadtxt(args.validation_input, delimiter='\t', comments=None, encoding='utf-8', dtype= str)
    num_epoch = int(args.num_epoch)
    learning_rate = float(args.learning_rate)
    
    embs_lst = [] 
    vector = np.vectorize(np.float_)
    for i in range(len(train_input)):
        embs = train_input[i][1:]
        embs[-1] = '1.0'
        embs_lst.append(vector(embs))
    
    embs_lst = np.asarray(embs_lst)

    test_embs_lst = []
    vector = np.vectorize(np.float_)
    for i in range(len(test_input)):
        embs = test_input[i][1:]
        embs[-1] = '1.0'
        test_embs_lst.append(vector(embs))

    test_embs_lst = np.asarray(test_embs_lst)

    val_embs_lst = []
    vector = np.vectorize(np.float_)
    for i in range(len(validation_input)):
        embs = validation_input[i][1:]
        embs[-1] = '1.0'
        val_embs_lst.append(vector(embs))
    val_embs_lst = np.asarray(val_embs_lst)


    train_label_lst = []
    for i in range(len(train_input)):
        train_labels = float(train_input[i][0])
        train_label_lst.append(train_labels)
    train_label_lst = np.asarray(train_label_lst)
    
    test_label_lst = []
    for i in range(len(test_input)):
        test_labels = float(test_input[i][0])
        test_label_lst.append(test_labels)

    val_label_lst = []
    for i in range(len(validation_input)):
        val_labels = float(validation_input[i][0])
        val_label_lst.append(val_labels)
    val_label_lst = np.asanyarray(val_label_lst)

    trained_weights = train(theta = np.zeros(embs_lst.shape[1]) , X = embs_lst, y = train_label_lst , num_epoch=num_epoch, learning_rate = learning_rate)

    train_y_hat_lst = predict(trained_weights , embs_lst) 
    test_y_hat_lst = predict(trained_weights, test_embs_lst)
    val_y_hat_lst = predict(trained_weights , val_embs_lst) 
    
    with open(args.metrics_out, 'w') as output:
        train_error = compute_error(train_label_lst, train_y_hat_lst)
        test_error = compute_error(test_label_lst, test_y_hat_lst ) 
        sent = "error(train): " + str(train_error) + "\n" +  "error(test): " + str(test_error)
        output.write(sent) 

    with open(args.train_out, 'w') as train_out:
        for i in range(len(train_y_hat_lst)):
            train_out.write(str(train_y_hat_lst[i])+'\n') 

    with open(args.test_out, 'w') as test_out:
        for i in range(len(test_y_hat_lst)):
            test_out.write(str(test_y_hat_lst[i])+'\n') 

    theta = np.zeros(embs_lst.shape[1])
    train_losses_1 = []
    train_losses_2 = []
    train_losses_3 = []
    val_losses = [] 

    for epoch in range(num_epoch):
        for i, x_i in enumerate(embs_lst):
            gradient = (sigmoid(np.dot(theta, x_i)) - train_label_lst[i]) * x_i
            theta -= 0.1 * gradient

        y_train_hat = sigmoid(np.dot(embs_lst, theta))
        train_loss = -np.mean(train_label_lst * np.log(y_train_hat) + (1 - train_label_lst) * np.log(1 - y_train_hat))
        train_losses_1.append(train_loss)

    theta = np.zeros(embs_lst.shape[1])
    for epoch in range(num_epoch):
        for i, x_i in enumerate(embs_lst):
            gradient = (sigmoid(np.dot(theta, x_i)) - train_label_lst[i]) * x_i
            theta -= 0.01 * gradient

        y_train_hat = sigmoid(np.dot(embs_lst, theta))
        train_loss_2 = -np.mean(train_label_lst * np.log(y_train_hat) + (1 - train_label_lst) * np.log(1 - y_train_hat))
        train_losses_2.append(train_loss_2)

    theta = np.zeros(embs_lst.shape[1])
    for epoch in range(num_epoch):
        for i, x_i in enumerate(embs_lst):
            gradient = (sigmoid(np.dot(theta, x_i)) - train_label_lst[i]) * x_i
            theta -= 0.001 * gradient

        y_train_hat = sigmoid(np.dot(embs_lst, theta))
        train_loss_3 = -np.mean(train_label_lst * np.log(y_train_hat) + (1 - train_label_lst) * np.log(1 - y_train_hat))
        train_losses_3.append(train_loss_3)


    plt.plot(range(num_epoch) , train_losses_1 , label = 'lr = 10^-1')
    plt.plot(range(num_epoch) , train_losses_2 , label = 'lr = 10^-2')
    plt.plot(range(num_epoch) , train_losses_3, label = 'lr = 10^-3')

    plt.xlabel('num of epoch')
    plt.ylabel('average negative log-likelihood')
    plt.title("average negative log-likelihood against epochs")
    plt.legend()
    plt.savefig('anll.png')
    plt.show()

    #python lr.py small_train.tsv small_val.tsv small_test.tsv small_train_labels.txt small_test_labels.txt small_metrics.txt 500 0.1 
    #python lr.py large_train_out.tsv large_val_out.tsv large_test_out.tsv large_train_labels.txt large_test_labels.txt large_metrics.txt 1000 0.1