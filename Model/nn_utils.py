import tensorflow as tf
import numpy as np

def initialize_embedding(shape,name='embedding'):

    if len(shape) == 1:
        value = np.zeros(shape)
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)

    emb = tf.Variable(initial_value=value,dtype=tf.float32,name=name)
    return emb

def build_fully_connected(in_dim,out_dim,input,activation='tanh'):
    raise NotImplemented()

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
        final_output = tf.reshape(tf.concat(state,axis=1),shape=[s[0],s[1],2*hid_dim])
    else:
        final_output = tf.concat(output, 2)

    return final_output