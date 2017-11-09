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
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, epsilon)
        def testing_phase():
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, epsilon)

        return tf.cond(training,training_phase,testing_phase)

class BNLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''
    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh', [x_size, 4 * self.num_units], initializer=tf.contrib.layers.xavier_initializer())
            W_hh = tf.get_variable('W_hh', [self.num_units, 4 * self.num_units], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm(xh, 'xh', self.training)
            bn_hh = batch_norm(hh, 'hh', self.training)

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(hidden, 4, 1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = batch_norm(new_c, 'c', self.training)

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zone-out
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell."""

    # zoneout_prob's variable must be a state of tuple (z_prob_cells,z_prob_state)
    def __init__(self, cell, zoneout_prob=(0.05,0.05), is_training=True, seed=None):
        self._cell = cell
        self._zoneout_prob = zoneout_prob
        self._seed = seed
        self.is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    # not it's for LSTM only
    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope)

        def training_phase():
            zip_data = zip(new_state, state, self._zoneout_prob)

            states = []
            for new_state_part, state_part, state_part_zoneout_prob in zip_data:
                cur_state = (1 - state_part_zoneout_prob) * tf.nn.dropout(new_state_part - state_part,
                                                                          (1 - state_part_zoneout_prob),
                                                                          seed=self._seed) + state_part
                states.append(cur_state)

            return tf.nn.rnn_cell.LSTMStateTuple(states[0], states[1])

        def testing_phase():
            zip_data = zip(new_state, state, self._zoneout_prob)

            states = []
            for new_state_part, state_part, state_part_zoneout_prob in zip_data:
                cur_state = state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                states.append(cur_state)

            return tf.nn.rnn_cell.LSTMStateTuple(states[0], states[1])

        zo_new_state = tf.cond(self.is_training,training_phase,testing_phase)

        return output, zo_new_state

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

