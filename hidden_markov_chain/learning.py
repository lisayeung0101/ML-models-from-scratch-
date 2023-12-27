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
    train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()

    # Initialize the initial, emission, and transition matrices
    # Increment the matrices
    # Add a pseudocount
    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter=" " for the matrices)
    
    #initial matrices (count the first state of every sequence)
    # print('train_data')
    # print(train_data)
    first_tag_dict = {k: 1 for k in tags_to_index.keys()}

    train_data = train_data[:10000]

    for i in range(len(train_data)):
        first_tag = train_data[i][0][1]
        first_tag_dict[first_tag] += 1
    total_count = sum(first_tag_dict.values())
    init_matrices = np.array([first_tag_dict[tag]/total_count for tag in first_tag_dict.keys()])
    # print(init_matrices)

    np.savetxt(init_out, init_matrices) 

    #emission matrices 
    num_states = len(tags_to_index)
    num_vocab = len(words_to_index)
    emit_matrices  = np.ones((num_states , num_vocab))

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            tag = train_data[i][j][1]
            # print('tag' , tag)
            word = train_data[i][j][0]
            # print('word' , word)
            tag_idx = tags_to_index[tag]
            word_idx = words_to_index[word]
            emit_matrices[tag_idx, word_idx] += 1

    # print(f'emit_matrices  is :  {emit_matrices}')

    count =  np.sum(emit_matrices ,axis = 1)
    count = count.reshape((1, num_states))
    # print ('count is ', count )
    emit_matrices = emit_matrices/ count.T
    # print(emit_matrices)      
    np.savetxt(emit_out, emit_matrices) 
 

    #transition matrices
    # states_dict = {k: 1 for k in tags_to_index.keys()} 
    trans_matrices = np.ones((num_states, num_states))
    for i in range(len(train_data)):
        for j in range(len(train_data[i]) - 1 ):
            seq_length = len(train_data[i])
            tag = train_data[i][j][1]
            tag_idx = tags_to_index[tag]
            next_tag = train_data[i][j + 1][1]
            next_tag_idx = tags_to_index[next_tag] 
            trans_matrices[tag_idx, next_tag_idx] += 1
    

    count =  np.sum(trans_matrices ,axis = 1)
    count = count.reshape((1, num_states))
    trans_matrices = trans_matrices/ count.T 
    np.savetxt(trans_out, trans_matrices)
