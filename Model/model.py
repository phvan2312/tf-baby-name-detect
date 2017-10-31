import tensorflow as tf
import numpy as np
from nn_utils import initialize_matrix, build_biRNN, crf_decode_with_batch, pad_common, pad_char


class NERModel:
    def __init__(self, id2char, id2word, id2label, id2pos, id2cap, id2reg,
                 char_emb_dim, word_emb_dim, cap_emb_dim, pos_emb_dim, reg_emb_dim,
                 char_hid_dim, word_hid_dim,
                 nn_for_char, dropout_prob, lr, optimize_method, clip,
                 dir_summary):
        self.id2char = id2char
        self.id2word = id2word
        self.id2label = id2label
        self.id2pos = id2pos
        self.id2cap = id2cap
        self.id2reg = id2reg  # assign None if we don't wanna use regex

        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.cap_emb_dim = cap_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.reg_emb_dim = reg_emb_dim

        self.char_hid_dim = char_hid_dim
        self.word_hid_dim = word_hid_dim

        self.nn_for_char = nn_for_char
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.optimize_method = optimize_method
        self.clip = clip

        self.dir_summary = dir_summary
        if tf.gfile.Exists(self.dir_summary):
            tf.gfile.DeleteRecursively(self.dir_summary)
        tf.gfile.MakeDirs(self.dir_summary)

    def __build_placeholder(self):
        self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None],
                                       name='char_ids')  # batch x max_len_sent x max_len_word
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')  # batch x max_len_sent
        self.cap_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='cap_ids')  # batch x max_len_sent
        self.pos_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos_ids')  # batch x max_len_sent
        self.reg_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='reg_ids')  # batch x mat_len_sent

        self.sentence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_length')  # bacth
        self.word_length = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                          name='word_length')  # batch x max_len_sent

        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    def __build_word_input(self):
        #
        # Initialize some embedding matrix
        #
        self.CHAR_EMB = initialize_matrix(shape=(len(self.id2char), self.char_emb_dim), name='character')
        self.WORD_EMB = initialize_matrix(shape=(len(self.id2word), self.word_emb_dim), name='word')
        self.CAP_EMB = initialize_matrix(shape=(len(self.id2cap), self.cap_emb_dim), name='cap')
        self.POS_EMB = initialize_matrix(shape=(len(self.id2pos), self.pos_emb_dim), name='pos')

        with tf.variable_scope('input_embedding'):
            self.char_enc = tf.nn.embedding_lookup(self.CHAR_EMB, self.char_ids)
            self.word_enc = tf.nn.embedding_lookup(self.WORD_EMB, self.word_ids)
            self.cap_enc = tf.nn.embedding_lookup(self.CAP_EMB, self.cap_ids)
            self.pos_enc = tf.nn.embedding_lookup(self.POS_EMB, self.pos_ids)

            #
            # Build word representation by concat all feature representation
            #
            word_from_char = self.__build_word_from_char()
            features = [self.word_enc, word_from_char, self.cap_enc, self.pos_enc]

            #
            # Using regex as an additional feature
            #
            if self.id2reg != None:
                self.REG_EMB = initialize_matrix(shape=(len(self.id2reg), self.reg_emb_dim), name='reg')
                self.reg_enc = tf.nn.embedding_lookup(self.REG_EMB, self.reg_ids)
                features.append(self.reg_enc)

            self.fn_word_enc = tf.concat(features, axis=2)

    # Build word representation from character level
    # We have two approach:
    # First, using Bi-LSTM for Char (default)
    # Second, using CNN
    def __build_word_from_char(self):
        def char_bilstm():
            return build_biRNN(input=self.char_enc, hid_dim=self.char_hid_dim, sequence_length=self.word_length,
                               cells=None, mode='character', scope='char_biRNN')

        def char_cnn():
            raise NotImplemented()

        if self.nn_for_char == 'bilstm':
            word_from_char = char_bilstm()
        elif self.nn_for_char == 'cnn':
            word_from_char = char_cnn()
        else:
            raise Exception('Model used for word representaion based on character-level you chose has'
                            ' not implemented yet.')

        return word_from_char

    def __build_sequence_tagging(self):
        with tf.variable_scope('model_loss'):
            #
            # Calculate loss for mode training
            #
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
                inputs=self.fn_scores, tag_indices=self.labels, sequence_lengths=self.sentence_length
            )

            self.train_loss = tf.reduce_mean(-log_likelihood)

            #
            # Calculate loss for mode evaluating (just for summarizer)
            #
            eval_log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(transition_params=self.transition, inputs=self.fn_scores,
                                                  tag_indices=self.labels, sequence_lengths=self.sentence_length)

            self.eval_loss = tf.reduce_mean(-eval_log_likelihood)

    def __build_optimizer(self):
        with tf.variable_scope('optimizer'):
            if self.optimize_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.optimize_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif self.optimize_method == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.lr)
            elif self.optimize_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.optimize_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise Exception('Ooptimizer method not be included yet')

            if self.clip > 0:
                grads, vars = zip(*optimizer.compute_gradients(self.train_loss))
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.clip)
                self.train_op = optimizer.apply_gradients(zip(clipped_grads, vars))
            else:
                self.train_op = optimizer.minimize(self.train_loss)

    def __build_fully_connected(self, in_dim, out_dim, input, activation, name='proj', scope='proj'):
        with tf.variable_scope(scope):

            setattr(self, 'W_%s' % name, initialize_matrix(shape=(in_dim, out_dim), name='w1_' + name))
            setattr(self, 'b_%s' % name, initialize_matrix(shape=(out_dim,), name='b_' + name))

            if activation == 'tanh':
                activate = tf.tanh
            elif activation == 'sigmoid':
                activate = tf.sigmoid
            elif activation == None:
                activate = lambda x: x
            else:
                raise Exception('Activation method not be implmented yet !!')

            s = tf.shape(input)
            s_lst = input.get_shape().as_list()

            W = getattr(self, 'W_%s' % name)
            b = getattr(self, 'b_%s' % name)

            input = tf.reshape(input, shape=(s[0] * s[1], s_lst[-1]))
            res = activate(tf.matmul(input, W) + b)

            return tf.reshape(res, shape=(s[0], s[1], out_dim))

    def __build_summary(self):
        tf.summary.scalar("loss", self.train_loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.dir_summary + '/train', self.sess.graph)
        self.eval_writer  = tf.summary.FileWriter(self.dir_summary + '/test')

    def build(self):
        #
        # initialize model and build word representation
        #
        self.__build_placeholder()
        self.__build_word_input()

        self.fn_word_enc = tf.nn.dropout(self.fn_word_enc, keep_prob=self.dropout)
        #
        # apply two bi-LSTM for word representation, then dropout some units
        #
        self.fn_w_v1 = build_biRNN(input=self.fn_word_enc, hid_dim=self.word_hid_dim,
                                   sequence_length=self.sentence_length,
                                   cells=None, mode='other', scope='word_biRNN_v1')

        self.fn_w_v2 = build_biRNN(input=self.fn_w_v1, hid_dim=self.word_hid_dim, sequence_length=self.sentence_length,
                                   cells=None, mode='other', scope='word_biRNN_v2')

        self.fn_output = tf.nn.dropout(self.fn_w_v2, keep_prob=self.dropout)

        #
        # fully connected layer to project from (2 x word_hid_dim) shape to (n_tag)
        #
        num_label = len(self.id2label)
        tanh_scores = self.__build_fully_connected(in_dim=2 * self.word_hid_dim, out_dim=self.word_hid_dim,
                                                   input=self.fn_output, activation='tanh', name='proj_v1',
                                                   scope='fully_connected')
        self.fn_scores = self.__build_fully_connected(in_dim=self.word_hid_dim, out_dim=num_label, input=tanh_scores,
                                                      activation=None, name='proj_v2', scope='fully_connected')

        #
        # Sequence level tagging (or CRF)
        #
        self.__build_sequence_tagging()

        #
        # Optimizer
        #
        self.__build_optimizer()

        #
        # Build session
        #
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        #
        # Build summary
        #
        self.__build_summary()

    def __create_feed_dict(self, batch, use_label=True, dropout_keep_prob=1.0):
        max_sentence_length = max(map(lambda x: len(x), [e['word_ids'] for e in batch]))

        ip_char_ids, word_length = pad_char(sequences=[e['char_ids'] for e in batch], pad_tok=0)
        ip_word_ids, sentence_length = pad_common(sequences=[e['word_ids'] for e in batch], pad_tok=0,
                                                  max_length=max_sentence_length)

        res = {
            self.char_ids: ip_char_ids,
            self.word_ids: ip_word_ids,
            self.cap_ids:
                pad_common(sequences=[e['cap_ids'] for e in batch], pad_tok=0, max_length=max_sentence_length)[0],
            self.pos_ids:
                pad_common(sequences=[e['pos_ids'] for e in batch], pad_tok=0, max_length=max_sentence_length)[0],
            self.sentence_length: sentence_length,
            self.word_length: word_length,
            self.dropout: dropout_keep_prob
        }

        if use_label:
            res[self.labels] = \
                pad_common(sequences=[e['label_ids'] for e in batch], pad_tok=0, max_length=max_sentence_length)[0]

        if self.id2reg != None:
            res[self.reg_ids] = \
                pad_common(sequences=[e['reg_ids'] for e in batch], pad_tok=0, max_length=max_sentence_length)[0]
        '''
        self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None],
                                       name='char_ids')  # batch x max_len_sent x max_len_word
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')  # batch x max_len_sent
        self.cap_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='cap_ids')  # batch x max_len_sent
        self.pos_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos_ids')  # batch x max_len_sent
        self.reg_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='reg_ids')  # batch x mat_len_sent

        self.sentence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_length')  # bacth
        self.word_length = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                          name='word_length')  # batch x max_len_sent

        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        '''
        return res

    def batch_run(self, batch, i, mode='train'):
        ip_feed_dict = self.__create_feed_dict(batch, dropout_keep_prob=self.dropout_prob)
        sentece_lengths = ip_feed_dict[self.sentence_length]

        if mode == 'train':
            _, loss, summary = self.sess.run([self.train_op, self.train_loss, self.merged],
                                             feed_dict=ip_feed_dict)

            self.train_writer.add_summary(summary, i)

            out = loss

        elif mode == 'eval':
            loss, scores, transition, summary = self.sess.run([self.eval_loss, self.fn_scores, self.transition, self.merged],
                                                     feed_dict=ip_feed_dict)
            predict_labels, _ = crf_decode_with_batch(scores=scores, sentence_lengths=sentece_lengths,
                                                                   transition=transition)
            self.eval_writer.add_summary(summary, i)
            out = predict_labels

        return out

    def close_writer(self):
        self.train_writer.close()
        self.eval_writer.close()


if __name__ == '__main__':
    pass
