import tensorflow as tf
#from nn_utils import orthogonal_initializer,bn_lstm_identity_initializer
import numpy as np

def batch_norm(inputs, name_scope, training, epsilon=1e-3, decay=0.99):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        def training_phase():
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, epsilon)
        def testing_phase():
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, epsilon)

        return tf.cond(training,training_phase,testing_phase)

class SelfAttention(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, memory, params, reuse=None):
        super(SelfAttention, self).__init__(_reuse=reuse)

        self.gru_cell = tf.contrib.rnn.GRUCell(num_units)
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
        X_s = tf.shape(X)
        if len(X.get_shape().as_list()) < 3:
            # X: [batch_size,m]
            # then expand dim
            X = tf.expand_dims(X, axis=1)
            X_s = tf.shape(X)

        # normalize W
        W_s = W.get_shape().as_list()
        out_dim = W_s[-1]

        if len(W_s) < 2:
            # W: [m,]
            # then reshape
            W = tf.reshape(W, shape=(-1, 1))
            out_dim = W.get_shape().as_list()[-1]

        X = tf.reshape(X, shape=(-1, X_s[-1]))
        res = tf.matmul(X, W)

        return tf.reshape(res, shape=(X_s[0], X_s[1], out_dim))

    def __call__(self, input, state,scope=None):
        # assume that num_units = 64, batch_size =32
        # inputs will have shape (32,64)
        # state will have shape (64,)
        with tf.variable_scope('attention_pool'):
            inp_enc = self.batch_matmul(input, self.W_inp_state)

            tanh = tf.tanh(inp_enc + self.mem_enc)

            a_t = self.batch_matmul(tanh, self.V)
            norm_a_t = tf.reshape(a_t, (tf.shape(a_t)[0], tf.shape(a_t)[1]) )
            a_t = tf.nn.softmax(norm_a_t, 1)

            c_t = tf.multiply(tf.expand_dims(a_t, axis=2), self.memory)
            c_t = tf.reduce_sum(c_t, axis=1)

            concat = tf.concat([input, c_t], axis=1)

            g_t = tf.matmul(concat, self.W_g)
            gru_inp = tf.multiply(tf.sigmoid(g_t),concat)

            return self.gru_cell(gru_inp, state)

