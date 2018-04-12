import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from .custom import batch_norm, SelfAttention

def batch_norm_layer(input, training, name='batch_norm'):
    ip_shape = tf.shape(input)
    ip_shape_lst = input.get_shape().as_list()

    if len(ip_shape_lst) > 2:
        input = tf.reshape(input, shape=(ip_shape[0] * ip_shape[1], ip_shape_lst[-1]))

    out = batch_norm(input, name, training)
    if len(ip_shape_lst) > 2:
        out = tf.reshape(out, shape=(ip_shape[0], ip_shape[1], ip_shape_lst[-1]))

    return out

def load_pretrained_word2vec(emb_size, id2word, pre_emb_path):
    vocab_size = len(id2word)

    # xavier initializer
    drange = np.sqrt(6. / (vocab_size + emb_size))
    W = drange * np.random.uniform(low=-1.0, high=1.0, size=(vocab_size, emb_size))

    # load pre-trained word2vec
    word2vec_model = KeyedVectors.load_word2vec_format(pre_emb_path, binary=True)

    # assign
    total_loaded = 0
    for key, word in list(id2word.items()):
        if word in word2vec_model.wv:
            W[key] = word2vec_model.wv[word]
            total_loaded += 1

    print(('-- Total loaded from pretrained:', total_loaded))
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

def build_char_cnn(input, filter_sizes, num_filter, scope):
    s_lst = input.get_shape().as_list()
    max_length_sentence = s_lst[1]
    max_length_word     = s_lst[2]
    emb_dim             = s_lst[3]

    with tf.variable_scope(scope):
        input = tf.reshape(input,shape=(-1,max_length_word, emb_dim))
        input = tf.expand_dims(input,3) # for corresponding with requirements in tensorflow

        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size,emb_dim,1,num_filter]

                # create variable
                W = tf.get_variable(name='W', shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name="b", shape=[num_filter], initializer=tf.zeros_initializer())

                conv = tf.nn.conv2d(
                    input,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_length_word - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filter * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, max_length_sentence, num_filters_total])

    return h_pool_flat

def build_bi_rnn(input, hid_dim, sequence_length, cells=None, mode='normal', scope='biRNN'):
    s = tf.shape(input=input)
    s_lst = input.get_shape().as_list()
    fn_input_dim = s_lst[-1]  # this value must be an integer.

    """
    expected:
    - input: [batch_size,max_sen_len,emb_dim]
        + for character: --> input: [batch_size,max_sen_len,max_word_len,emb_dim], must be reshaped into 
        [batch_size * max_sen_len, max_word_len, emb_dim]
    - sequence_length: [batch_size]
        + for character: --> input: [batch_size,max_sen_len], must be reshaped into [batch_size * max_sen_len]
    - hid_dim: integer
    - cells: tuple (RNNCell, RNNCell), or set None to be reinitialized
    - mode: ['character'] or anything else
    - 
    """

    with tf.variable_scope(scope):
        if cells is not None: cell_fw, cell_bw = cells
        else:
            cell_fw = tf.contrib.rnn.LSTMCell(hid_dim, initializer=tf.contrib.layers.xavier_initializer())
            cell_bw = tf.contrib.rnn.LSTMCell(hid_dim, initializer=tf.contrib.layers.xavier_initializer())

        if len(s_lst) > 3:  # character mode (bi-lstm for character-level)
            input = tf.reshape(input, shape=(s[0] * s[1], s[2], fn_input_dim))
            sequence_length = tf.reshape(sequence_length, shape=(s[0] * s[1],))

        """
        outputs: (batch_size, max_sen_len, hidden_size) for each forward/backward
        state: (batch_sizem hidden_size) for each forward/backward
        """

        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,dtype=tf.float32,
                                                        inputs=input, sequence_length=sequence_length)

        if mode == 'character':
            final_output = tf.reshape(tf.concat(state[:][1], axis=1), shape=(s[0], s[1], 2 * hid_dim))
        else:
            final_output = tf.concat(outputs, axis=2)

        return final_output

# build a self-matching attention
# get intuition from this page:https://yerevann.github.io/2017/08/25/challenges-of-reproducing-r-net-neural-network-using-keras/
def build_self_attention(input, hid_dim, sequence_length, scope):
    # self-matching attention
    with tf.variable_scope(scope):
        W_mem_state = initialize_matrix(name='W_mem_state',shape=(hid_dim,hid_dim))
        W_inp_state = initialize_matrix(name='W_inp_state',shape=(hid_dim,hid_dim))
        V = initialize_matrix(name='V',shape=(hid_dim,1))
        W_g = initialize_matrix(name='W_g',shape=(2*hid_dim,2*hid_dim))

        params = {
            'W_mem_state' : W_mem_state,
            'W_inp_state' : W_inp_state,
            'V' : V,
            'W_g' : W_g,
        }

        cell_fw = SelfAttention(num_units=hid_dim,memory=input,params=params)
        output, _  = tf.nn.dynamic_rnn(cell=cell_fw,inputs=input,sequence_length=sequence_length,dtype=tf.float32)

        return output

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
def pad_char(sequences, pad_tok, max_length_word, max_length_sentence):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    #max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        # all words are same length now
        sp, sl = pad_common(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    sequence_padded, _ = pad_common(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = pad_common(sequence_length, 0, max_length_sentence)

    return sequence_padded, sequence_length
