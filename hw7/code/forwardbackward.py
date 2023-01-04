import argparse
import numpy as np
# import matplotlib.pyplot as plt

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
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(v):
        m = np.amax(v, axis=1)
        return np.log((np.sum(np.exp(v-m[:, np.newaxis]), axis=1))) + m

def forwardbackward(seq, loginit, logtrans, logemit):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in
    log_alpha = np.zeros((L, M))
    log_alpha[0, :] = loginit + logemit[:, seq[0]]

    for i in range(1, L):
        w_ind = seq[i]
        log_alpha[i, :] = logemit[:, w_ind] + logsumexp(log_alpha[i-1, :] + logtrans.T)

    # Initialize log_beta and fill it in
    log_beta = np.zeros((L, M))
    for i in range(L-2, -1, -1): 
        w_ind = seq[i+1]
        log_beta[i, :] = logsumexp(logemit[:, w_ind] + log_beta[i+1, :] + logtrans)

    # Compute the predicted tags for the sequence
    Y_t = log_alpha + log_beta
    Y_t_hat = np.argmax(Y_t, axis=1)

    # Compute the log-probability of the sequence
    log_prob = logsumexp(log_alpha[-1, :][np.newaxis, :])[0]

    # Return the predicted tags and the log-probability
    # pass
    return Y_t_hat, log_prob
    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    loginit, logtrans, logemit = np.log(hmminit), np.log(hmmtrans), np.log(hmmemit)
    
    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.

    N = len(validation_data)
    tags = []
    preds, log_probs = [], []
    for i in range(N):
        sentence = validation_data[i]
        M = len(sentence)
        seq, tag = [], []
        for j in range(M):
            w, t = sentence[j]
            w_ind, t_ind = words_to_indices[w], tags_to_indices[t]
            seq.append(w_ind)
            tag.append(t_ind)
        pred, log_prob = forwardbackward(seq, loginit, logtrans, logemit)
        preds.append(pred)
        log_probs.append(log_prob)
        tags.append(tag)

    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    avg_logprob = sum(log_probs)/N
    acc = 0
    n_tag = 0
    for i in range(N):
        acc += sum(tags[i]==preds[i])  
        n_tag += len(tags[i])
    acc = acc/n_tag
    
    with open(predicted_file, "w") as f:
        for i in range(N):
            sentence = validation_data[i]
            pred = preds[i]
            M = len(sentence)
            for j in range(M):
                f.write(str(sentence[j][0]) + "\t")
                pred_tag = list(tags_to_indices.keys())[list(tags_to_indices.values()).index(pred[j])]
                f.write(str(pred_tag) + "\n")
            f.write("\n")
    f.close()

    file = open(metric_file,"w")
    lines = [f'Average Log-Likelihood: {avg_logprob}\n',f'Accuracy: {acc}\n']
    file.writelines(lines)
    file.close()


    # pass

    # # Get the input data
    # validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    # N_list = [10, 100, 1000, 10000]
    # val_probs = []
    # train_probs = []
    # for N in N_list:
    #     hmminit = np.loadtxt(f"en_data/hmminit_{N}.txt", dtype=float, delimiter=" ")
    #     hmmemit = np.loadtxt(f"en_data/hmmemit_{N}.txt", dtype=float, delimiter=" ")
    #     hmmtrans = np.loadtxt(f"en_data/hmmtrans_{N}.txt", dtype=float, delimiter=" ")
    #     loginit, logtrans, logemit = np.log(hmminit), np.log(hmmtrans), np.log(hmmemit)

    #     train_data = list()
    #     with open("en_data/train.txt", "r") as f:
    #         examples = f.read().strip().split("\n\n")
    #         for example in examples:
    #             xi = [pair.split("\t") for pair in example.split("\n")]
    #             train_data.append(xi)
        
    #     # For each sequence, run forward_backward to get the predicted tags and 
    #     # the log-probability of that sequence.

    #     N = len(validation_data)
    #     tags = []
    #     preds, log_probs = [], []
    #     for i in range(N):
    #         sentence = validation_data[i]
    #         M = len(sentence)
    #         seq, tag = [], []
    #         for j in range(M):
    #             w, t = sentence[j]
    #             w_ind, t_ind = words_to_indices[w], tags_to_indices[t]
    #             seq.append(w_ind)
    #             tag.append(t_ind)
    #         pred, log_prob = forwardbackward(seq, loginit, logtrans, logemit)
    #         preds.append(pred)
    #         log_probs.append(log_prob)
    #         tags.append(tag)

    #     val_probs.append(sum(log_probs)/N)

    #     N = len(train_data)
    #     tags = []
    #     preds, log_probs = [], []
    #     for i in range(N):
    #         sentence = train_data[i]
    #         M = len(sentence)
    #         seq, tag = [], []
    #         for j in range(M):
    #             w, t = sentence[j]
    #             w_ind, t_ind = words_to_indices[w], tags_to_indices[t]
    #             seq.append(w_ind)
    #             tag.append(t_ind)
    #         pred, log_prob = forwardbackward(seq, loginit, logtrans, logemit)
    #         preds.append(pred)
    #         log_probs.append(log_prob)
    #         tags.append(tag)

    #     train_probs.append(sum(log_probs)/N)
    # print(train_probs)
    # print(val_probs)
    # plt.figure(figsize=(10, 8))
    # plt.plot([np.log(10), np.log(100), np.log(1000), np.log(10000)], train_probs, '-b')
    # plt.plot([np.log(10), np.log(100), np.log(1000), np.log(10000)], val_probs, '-g')
    # plt.xticks([np.log(10), np.log(100), np.log(1000), np.log(10000)])
    # plt.xlabel('number of first sequences to learn learn HMM arameters (in log-scale)')
    # plt.ylabel('average log likelihood')
    # plt.title('average log likelihood vs. using different number of first sequences to learn HMM arameters')
    # plt.legend(['average log likelihood for train.txt', 'average log likelihood for validation.txt'])
    # plt.savefig('5.png')


        

