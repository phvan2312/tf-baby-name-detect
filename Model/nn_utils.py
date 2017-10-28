import tensorflow as tf
import numpy as np

def initialize_matrix(shape, mode='xavier', name='embedding'):
    if len(shape) == 1:
        initializer = tf.zeros_initializer # for bias
    elif mode == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer
    else:
        initializer = tf.truncated_normal_initializer

    emb = tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=initializer())

    return emb

def build_biRNN(input,hid_dim,sequence_length,cells=None,mode='normal'):
    s = input.get_shape().as_list()

    if cells is not None:
        cell_fw, cell_bw = cells
    else:
        cell_fw = tf.contrib.rnn.LSTMCell(hid_dim)
        cell_bw = tf.contrib.rnn.LSTMCell(hid_dim)

    if len(s) > 3: # for character
        input = tf.reshape(input,shape=[s[0]*s[1],s[2],-1])
        sequence_length = tf.reshape(sequence_length,shape=(s[0]*s[1],))

    output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                    inputs=input, sequence_length=sequence_length,
                                                    dtype=tf.float32)

    if mode == 'character':
        final_output = tf.reshape(tf.concat(state[:][1],axis=1),shape=[s[0],s[1],2*hid_dim])
    else:
        final_output = tf.concat(output, axis=2)

    return final_output

# params sentence_lengths contain integer value which is actual
# length of each post in batch.
def crf_decode_with_batch(scores,sentence_lengths,transition):
    predict_labels = []
    predict_scores = []

    for score, sentence_length in zip(scores,sentence_lengths):
        norm_score = score[:sentence_length]
        predict_label, predict_score = tf.contrib.crf.viterbi_decode(norm_score, transition)
        predict_labels.append(predict_label)
        predict_scores.append(predict_score)

    return predict_labels, predict_scores