import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


def load_pretrained_word2vec(emb_size, id2word, pre_emb_path):
    vocab_size = len(id2word)

    # xavier initializer
    drange = np.sqrt(6. / (vocab_size + emb_size))
    W = drange * np.random.uniform(low=-1.0, high=1.0, size=(vocab_size, emb_size))

    # load pretrained word2vec
    word2vec_model = KeyedVectors.load_word2vec_format(pre_emb_path, binary=True)

    # assign
    total_loaded = 0
    for key, word in id2word.items():
        if word in word2vec_model.wv:
            W[key] = word2vec_model.wv[word]
            total_loaded += 1

    print ('-- Total loaded from pretrained:', total_loaded)
    return W


def initialize_matrix(shape, mode='xavier', name='embedding'):
    if len(shape) == 1:
        initializer = tf.zeros_initializer  # for bias
    elif mode == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer
    else:
        initializer = tf.truncated_normal_initializer

    emb = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=initializer())

    return emb


def build_biRNN(input, hid_dim, sequence_length, cells=None, mode='normal', scope='biRNN'):
    s = tf.shape(input=input)
    s_lst = input.get_shape().as_list()
    fn_input_dim = s_lst[-1]  # this value must exactly be an integer.

    with tf.variable_scope(scope):
        if cells is not None:
            cell_fw, cell_bw = cells
        else:
            cell_fw = tf.contrib.rnn.LSTMCell(hid_dim)
            cell_bw = tf.contrib.rnn.LSTMCell(hid_dim)

        if len(s_lst) > 3:  # for character
            input = tf.reshape(input, shape=(s[0] * s[1], s[2], fn_input_dim))
            sequence_length = tf.reshape(sequence_length, shape=(s[0] * s[1],))

        output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                        inputs=input, sequence_length=sequence_length,
                                                        dtype=tf.float32)

        if mode == 'character':
            final_output = tf.reshape(tf.concat(state[:][1], axis=1), shape=(s[0], s[1], 2 * hid_dim))
        else:
            final_output = tf.concat(output, axis=2)

        return final_output


# params sentence_lengths contain integer value which is actual
# length of each post in batch.
def crf_decode_with_batch(scores, sentence_lengths, transition):
    predict_labels = []
    predict_scores = []

    for score, sentence_length in zip(scores, sentence_lengths):
        norm_score = score[:sentence_length]
        predict_label, predict_score = tf.contrib.crf.viterbi_decode(norm_score, transition)
        predict_labels.append(predict_label)
        predict_scores.append(predict_score)

    return predict_labels, predict_scores


# code from : https://github.com/guillaumegenthial/sequence_tagging
# original method name : _pad_sequence
def pad_common(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


# code from : https://github.com/guillaumegenthial/sequence_tagging
# original method name : pad_sequence
def pad_char(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        # all words are same length now
        sp, sl = pad_common(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = pad_common(sequence_padded,
                                    [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = pad_common(sequence_length, 0,
                                    max_length_sentence)

    return sequence_padded, sequence_length
