from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import RNNCell

class Attention_GRUCell(RNNCell):
    def __init__(self,num_units, memory, params, reuse=None):
        super(Attention_GRUCell, self).__init__(_reuse=reuse)

        self.rnn_cell = tf.contrib.rnn.GRUCell(num_units)
        self.num_units = num_units
        self.memory = memory

        self.W_mem_state = params['W_mem_state']
        self.W_inp_state = params['W_inp_state']
        self.V = params['V']
        self.W_g = params['W_g']

        # dot(mem,W_mem_state) can be calculated statically
        self.mem_enc = self.batch_matmul(self.memory, self.W_mem_state)

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def batch_matmul(self, X, W):
        '''
        Customizer matmul function of tensorflow
        X: [batch_size,n,m],
        W: [m,p]
        Result: [batch_size,n,p]
        '''
        # normalize X
        X_s = X.get_shape().as_list()
        if len(X_s) < 3 :
            # X: [batch_size,m]
            # then expand dim
            X = tf.expand_dims(X,axis=1)
            X_s = X.get_shape().as_list()

        # normalize W
        W_s = W.get_shape().as_list()
        if len(W_s) < 2:
            # W: [m,]
            # then reshape
            W = tf.reshape(W,shape=(-1,1))
            W_s = W.get_shape().as_list()

        X = tf.reshape(X,shape=(-1,X_s[-1]))
        res = tf.matmul(X,W)

        return tf.reshape(res,shape=(X_s[0],X_s[1],W_s[-1]))

    def call(self, input, state):
        # assume that num_units = 64, batch_size =32
        # inputs will have shape (32,64)
        # state will have shape (64,)
        with vs.variable_scope('attention_pool'):
            inp_enc = self.batch_matmul(input, self.W_inp_state)

            tanh = tf.tanh(inp_enc + self.mem_enc)

            a_t = self.batch_matmul(tanh, self.V)
            a_t = tf.nn.softmax(tf.squeeze(a_t),1)

            c_t = tf.multiply(tf.expand_dims(a_t,axis=2),self.memory)
            c_t = tf.reduce_sum(c_t,axis=1)

            concat = tf.concat([input,c_t],axis=1)

            g_t = tf.matmul(concat,self.W_g)
            rnn_inp = tf.sigmoid(g_t) * concat

            return self.rnn_cell(rnn_inp,state)
