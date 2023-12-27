import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("training_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)

    training_data = list()
    with open(args.training_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            training_data.append(xi)

    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data,training_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix


def logsumexp(matrix):
    # If input is a vector, convert it to a row vector
    if len(matrix.shape) == 1:
        matrix = matrix[np.newaxis, :]
    # Apply log-sum-exp trick to each row of the matrix
    row_max = np.max(matrix, axis=1, keepdims=True)
    return row_max.squeeze() + np.log(np.sum(np.exp(matrix - row_max), axis=1))


def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix

        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

    compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in 
    log_alpha = np.zeros((M,L))
    log_alpha[:, 0] = loginit + logemit[:, words_to_indices[seq[0]]]
    for t in range(1, L):
        for j in range(M): 
            log_alpha[j, t] = logemit[j , words_to_indices[seq[t]]] + logsumexp(logtrans[:, j].T  + log_alpha[: ,t - 1 ].T)

    # Initialize log_beta and fill it in
    log_beta = np.zeros((M,L))
    for t in range( L-2 , -1 , -1):
        for j in range(M):
            log_beta[j, t] = logsumexp(logemit[:, words_to_indices[seq[t + 1]]]  + logtrans[j, :] + log_beta[:, t+1]) 

    # Compute the predicted tags for the sequence 
    predicted_tag = [] 
    for t in range(L):
        probs = log_alpha[:, t] + log_beta[:, t]
        tag_idx = np.argmax(probs)
        tag_list = [tag for tag, idx in tags_to_indices.items() if idx == tag_idx]
        for i in range(len(tag_list)): 
            predicted_tag.append(tag_list[i])


    # Compute the stable log-probability of the sequence
    log_prob_seq = logsumexp(log_alpha[:, -1])

    # Return the predicted tags and the log-probability
    return predicted_tag , log_prob_seq
    
    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, training_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs() 

    loginit = np.log(hmminit)
    logtrans = np.log(hmmtrans) 
    logemit = np.log(hmmemit)

    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.   

    val_seq = []
    log_prob_lst = []
    pred_tag_lst = []
    val_tag_lst = []
    word_all_lst = [] 
    pred_tag_out = []
    word_out = []
    val_log_prob_lst = []
    train_log_prob_lst = []

    # validation_data_10 = validation_data[:10]
    # validation_data_100 = validation_data[:100]
    # validation_data_1000 = validation_data[:1000]
    # validation_data_10000 = validation_data[:10000]
    # val_set = [validation_data_10 , validation_data_100, validation_data_1000 ,validation_data_10000 ]

    # training_data_10 = training_data[:10]
    # training_data_100 = training_data[:100]
    # training_data_1000 = training_data[:1000]
    # training_data_10000 = training_data[:10000]
    # train_set = [training_data_10 , training_data_100, training_data_1000 ,training_data_10000 ] 

    # for val in val_set:
    for i in range(len(validation_data)):
        each_seq_pair = validation_data[i]
        val_seq = [item[0] for item in each_seq_pair]
        predicted_tags , log_prob = forwardbackward(val_seq , loginit, logtrans, logemit, words_to_indices, tags_to_indices)
        pred_tag_lst.extend(predicted_tags)
        # pred_tag_out.append(predicted_tags)
        val_tag_lst.extend([item[1] for item in each_seq_pair])
        word_all_lst.extend([item[0] for item in each_seq_pair])
        # word_out.append([item[0] for item in each_seq_pair]) 
        val_log_prob_lst.append(log_prob)


    # for train in train_set:
    for i in range(len(training_data)):
        each_seq_pair = training_data[i]
        val_seq = [item[0] for item in each_seq_pair]
        predicted_tags , log_prob = forwardbackward(val_seq , loginit, logtrans, logemit, words_to_indices, tags_to_indices)
        pred_tag_lst.extend(predicted_tags)
        # pred_tag_out.append(predicted_tags)
        val_tag_lst.extend([item[1] for item in each_seq_pair])
        word_all_lst.extend([item[0] for item in each_seq_pair])
        # word_out.append([item[0] for item in each_seq_pair]) 
        train_log_prob_lst.append(log_prob)


    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.

    # print('length of ll for validation set : ')
    # print(len(val_log_prob_lst))


    # print(f'length of complete validation data {len(validation_data)}')

    # val_avg_ll_10 = np.mean(val_log_prob_lst[:10])
    # val_avg_ll_100 = np.mean(val_log_prob_lst[10:110])
    # val_avg_ll_1000 = np.mean(val_log_prob_lst[110:1110])
    # val_avg_ll_10000 = np.mean(val_log_prob_lst[1110:])
    # print (f'val_avg_ll_10 {val_avg_ll_10}')
    # print (f'length of val_avg_ll_10 {len(val_log_prob_lst[:10])}')
    # print (f'val_avg_ll_100 {val_avg_ll_100}')
    # print (f'length of val_avg_ll_100 {len(val_log_prob_lst[10:110])}')
    # print (f'val_avg_ll_1000 {val_avg_ll_1000}')
    # print (f'length of val_avg_ll_1000 {len(val_log_prob_lst[110:1110])}')
    # print (f'val_avg_ll_10000 {val_avg_ll_10000}')
    # print (f'length of val_avg_ll_10000 {len(val_log_prob_lst[1110:])}')


    # train_avg_ll_10 = np.mean(train_log_prob_lst[:10])
    # train_avg_ll_100 = np.mean(train_log_prob_lst[10:110])
    # train_avg_ll_1000 = np.mean(train_log_prob_lst[110:1110])
    # train_avg_ll_10000 = np.mean(train_log_prob_lst[1110:])
    # print (f'train_avg_ll_10 {train_avg_ll_10}')
    # print (f'length of train_avg_ll_10 {len(train_log_prob_lst[:10])}')
    # print (f'train_avg_ll_100 {train_avg_ll_100}')
    # print (f'length of train_avg_ll_100 {len(train_log_prob_lst[10:110])}')
    # print (f'train_avg_ll_1000 {train_avg_ll_1000}')
    # print (f'length of train_avg_ll_1000 {len(train_log_prob_lst[110:1110])}')
    # print (f'train_avg_ll_10000 {train_avg_ll_10000}')
    # print (f'length of train_avg_ll_10000 {len(train_log_prob_lst[1110:])}')


    # correct_count = 0 
    # for i in range(len(pred_tag_lst)):
    #     if pred_tag_lst[i] == val_tag_lst[i]: 
    #         correct_count += 1
    # accuracy = correct_count / len(pred_tag_lst)

    val_avg_ll = np.mean(val_log_prob_lst)
    train_avg_ll = np.mean(train_log_prob_lst) 

    with open(metric_file, 'w') as output:
        sent = "Average Log-Likelihood: " + str(val_avg_ll) + "\n" +  "train_avg_ll: " + str(train_avg_ll)
        output.write(sent)
    
    # import matplotlib.pyplot as plt
    # num_seq = [10, 100, 1000, 10000]
    # val_loss = [ -60.15367407859437, -60.10468709227909, -60.09553680464696 , -60.087579647244404 ]
    # train_loss = [ -59.92557253515608, -59.84003486217851, -59.823857492084606 , -59.82262116598064] 
    # plt.xscale("log")
    # plt.plot(num_seq,val_loss, label = "average loglikelihood in validation set")
    # plt.plot(num_seq, train_loss, label = "average loglikelihood in training set")
    # plt.xlabel('# of Sequences')
    # plt.ylabel('average log-likelihood')
    # plt.title("average log-likelihood against number of sequences")
    # plt.legend()
    # plt.savefig('plot.png')



    # print(f'pred tag out  {pred_tag_out}')
    # print(f'word out  {word_out}')
    # sent = ''
    # with open(predicted_file, 'w') as pred_out:
    #     for i in range(len(pred_tag_out)):
    #         for j in range(len(pred_tag_out[i])):
    #             sent += (str(word_out[i][j]) + "\t"  +(str(pred_tag_out[i][j]) + "\n"))
    #         sent += "\n"
            
    #     pred_out.write(sent)
    
