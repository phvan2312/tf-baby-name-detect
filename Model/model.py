import tensorflow as tf
from .nn_utils import initialize_matrix, build_bi_rnn, crf_decode_with_batch, pad_common, pad_char, \
    load_pretrained_word2vec, batch_norm_layer, build_self_attention, build_char_cnn


class NERModel:
    def __init__(self, id2char, id2word, id2label, id2pos, id2cap, id2reg,
                 char_emb_dim, word_emb_dim, cap_emb_dim, pos_emb_dim, reg_emb_dim,
                 char_hid_dim, word_hid_dim,
                 nn_for_char, dropout_prob, lr, optimize_method, clip,
                 dir_summary, pre_emb_path,
                 max_length_word, max_length_sentence,
                 filter_sizes, num_filter,
                 use_char,use_pos,use_cap,use_reg):

        tf.reset_default_graph()

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

        self.filter_sizes = filter_sizes
        self.num_filter = num_filter

        self.nn_for_char = nn_for_char
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.optimize_method = optimize_method
        self.clip = clip

        self.dir_summary = dir_summary
        if tf.gfile.Exists(self.dir_summary): tf.gfile.DeleteRecursively(self.dir_summary)
        tf.gfile.MakeDirs(self.dir_summary)

        self.np_WORD_EMB = load_pretrained_word2vec(emb_size=self.word_emb_dim, id2word=self.id2word,
                                                    pre_emb_path=pre_emb_path) if pre_emb_path != '' else None
        self.freq_equal_summary = 10

        self.max_length_word = max_length_word
        self.max_length_sentence = max_length_sentence

        self.use_char = use_char
        self.use_cap  = use_cap
        self.use_pos  = use_pos
        self.use_reg  = use_reg

    # Build word representation from character level
    # We have two approach:
    # First, using Bi-LSTM for Char (default)
    # Second, using CNN
    def __build_word_from_char(self):
        def char_bilstm():
            return build_bi_rnn(input=self.char_enc, hid_dim=self.char_hid_dim, sequence_length=self.word_length,
                                cells=None, mode='character', scope='char_bidirection')

        def char_cnn():
            return build_char_cnn(input=self.char_enc, filter_sizes=self.filter_sizes, num_filter=self.num_filter,
                                  scope='char_cnn')

        if self.nn_for_char == 'bilstm': word_from_char = char_bilstm()
        elif self.nn_for_char == 'cnn': word_from_char = char_cnn()
        else:
            raise Exception('Model used for word representaion based on character-level you chose has'
                            ' not implemented yet.')

        return word_from_char

    def __build_sequence_tagging(self):
        """
        calculate loss for mode training
        """
        log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
            inputs=self.fn_scores, tag_indices=self.labels, sequence_lengths=self.sentence_length
        )

        self.train_loss = tf.reduce_mean(-log_likelihood)

        """
        calculate loss for mode evaluating (just for summarizer)
        """
        eval_log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(transition_params=self.transition,
                                                                   inputs=self.fn_scores,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sentence_length)

        self.eval_loss = tf.reduce_mean(-eval_log_likelihood)

    def __build_optimizer(self):
        if self.optimize_method == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr_placeholder)
        elif self.optimize_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lr_placeholder)
        elif self.optimize_method == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr_placeholder)
        elif self.optimize_method == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.lr_placeholder)
        elif self.optimize_method == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.lr_placeholder)
        else:
            raise Exception('Optimizer method not be included yet')

        if self.clip > 0:
            grads, vars = list(zip(*optimizer.compute_gradients(self.train_loss)))
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.clip)
            self.train_op = optimizer.apply_gradients(list(zip(clipped_grads, vars)))
        else:
            self.train_op = optimizer.minimize(self.train_loss)

    def __build_fully_connected(self, in_dim, out_dim, input, activation, name='dense', batch_norm=False,
                                is_training=None):

        setattr(self, 'W_%s' % name, initialize_matrix(shape=(in_dim, out_dim), name='w1_' + name))
        setattr(self, 'b_%s' % name, initialize_matrix(shape=(out_dim,), name='b_' + name))

        if activation == 'tanh':
            activate = tf.tanh
        elif activation == 'sigmoid':
            activate = tf.sigmoid
        elif activation == None:
            activate = lambda x: x
        else:
            raise Exception('Activation method not be implemented yet !!')

        s = tf.shape(input)
        s_lst = input.get_shape().as_list()

        W = getattr(self, 'W_%s' % name)
        b = getattr(self, 'b_%s' % name)

        if len(s_lst) > 2:
            input = tf.reshape(input, shape=(s[0] * s[1], s_lst[-1]))

        z = tf.matmul(input, W) + b

        if batch_norm:
            if is_training == False:
                raise Exception('Invalid argument is_training(must be True or False)')
            z = batch_norm_layer(z, training=is_training, name='batch_norm_' + name)

        a = activate(z)

        return tf.reshape(a, shape=(s[0], s[1], out_dim)) if len(s_lst) > 2 else a

    def __build_summary(self):
        #### LOSS #####
        # build general loss (because for training phase we need train_loss and otherwise eval_loss.)
        mode = tf.cast(self.is_training,dtype=tf.float32) # is training or evaling
        loss = mode * (self.train_loss) + (1 - mode)*self.eval_loss

        tf.summary.scalar("loss", loss)

        self.merged = tf.summary.merge_all()

    def build(self):

        with tf.variable_scope('embedding'):
            """
            initialize model and build word representation
            """
            self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length_sentence, self.max_length_word],
                                           name='char_ids')  # batch x max_len_sent x max_len_word
            self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length_sentence],
                                           name='word_ids')  # batch x max_len_sent
            self.cap_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length_sentence],
                                          name='cap_ids')  # batch x max_len_sent
            self.pos_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length_sentence],
                                          name='pos_ids')  # batch x max_len_sent
            self.reg_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length_sentence],
                                          name='reg_ids')  # batch x mat_len_sent

            self.sentence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_length')  # bacth
            self.word_length = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length_sentence],
                                              name='word_length')  # batch x max_len_sent

            self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length_sentence], name='labels')
            self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
            self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")

            self.lr_placeholder = tf.placeholder_with_default(input=self.lr, shape=[], name='lr')

            """
            embedding input
            """

            if self.np_WORD_EMB is not None:
                self.WORD_EMB = tf.get_variable("word_emb", shape=(len(self.id2word), self.word_emb_dim),
                                                initializer=tf.constant_initializer(self.np_WORD_EMB))
            else:
                self.WORD_EMB = initialize_matrix(shape=(len(self.id2word), self.word_emb_dim), name='word_emb')

            self.CHAR_EMB = initialize_matrix(shape=(len(self.id2char), self.char_emb_dim), name='char_emb')
            self.CAP_EMB  = initialize_matrix(shape=(len(self.id2cap), self.cap_emb_dim), name='cap_emb')
            self.POS_EMB  = initialize_matrix(shape=(len(self.id2pos), self.pos_emb_dim), name='pos_emb')
            self.REG_EMB  = initialize_matrix(shape=(len(self.id2reg), self.reg_emb_dim), name='reg_emb')

            self.char_enc = tf.nn.embedding_lookup(self.CHAR_EMB, self.char_ids,name='char_enc')
            self.word_enc = tf.nn.embedding_lookup(self.WORD_EMB, self.word_ids,name='word_enc')
            self.cap_enc  = tf.nn.embedding_lookup(self.CAP_EMB, self.cap_ids,name='cap_enc')
            self.pos_enc  = tf.nn.embedding_lookup(self.POS_EMB, self.pos_ids,name='pos_enc')
            self.reg_enc  = tf.nn.embedding_lookup(self.REG_EMB, self.reg_ids,name='reg_enc')

            """
            build word by using character information by concat all feature representation
            """
            word_from_char = self.__build_word_from_char()

            """
            build final word encoding by concatenating all information.
            """
            add_features    = [self.word_enc, word_from_char, self.cap_enc, self.pos_enc,self.reg_enc]
            is_used         = [1, self.use_char, self.use_cap, self.use_pos, self.use_reg]

            chosen_features = [v for (v,c) in zip(add_features,is_used) if c]

            self.fn_word_enc = tf.concat(chosen_features, axis=2)

            """
            apply some regularization terms
            """
            self.fn_word_enc = batch_norm_layer(self.fn_word_enc, training=self.is_training, name='bach_norm_word')
            self.fn_word_enc = tf.nn.dropout(self.fn_word_enc, keep_prob=self.dropout)

        with tf.variable_scope('attention_bi_lstm'):
            """
            bidirectional-LSTM and self attention
            """
            self.fn_w_v1 = build_bi_rnn(input=self.fn_word_enc, hid_dim=self.word_hid_dim, sequence_length=self.sentence_length,
                                        cells=None, mode='other', scope='word_bidirection_lstm_v1')

            self.fn_w_v1 = batch_norm_layer(self.fn_w_v1, training=self.is_training, name='batch_norm_lstm1')

            # self.fn_w_v2 = build_biRNN(input=self.fn_w_v1, hid_dim=self.word_hid_dim,
            #                            sequence_length=self.sentence_length,cells=None, mode='other',
            #                            scope='word_bidirection_lstm_v2')

            # Note that: Attention consumes so much memory and computation, so i decided to decrease word hidden embedding from 100 downto 50.
            # But it still take about 2 hours to finish training phase (~1000 samples).
            # If you don't wanna use this, so turn word hidden embedding back to 100, because this is the best value i found for previous configs.
            self.fn_w_v2 = build_self_attention(input=self.fn_w_v1, hid_dim=2 * self.word_hid_dim,
                                                sequence_length=self.sentence_length, scope='self_attention')

        #self.fn_output = batch_norm_layer(self.fn_w_v2, training=self.is_training, name='batch_norm_lstm2')

        self.fn_output = self.fn_w_v2 #self.fn_w_v2

        with tf.variable_scope('fully_connected'):
            """
            fully connected layer to project from (2 x word_hid_dim) shape to (n_tag)
            """
            num_label = len(self.id2label)

            tanh_scores = self.__build_fully_connected(in_dim=2 * self.word_hid_dim, out_dim=self.word_hid_dim,
                                                       input=self.fn_output, activation='tanh', name='dense_v1')
            self.fn_scores = self.__build_fully_connected(in_dim=self.word_hid_dim, out_dim=num_label,
                                                          input=tanh_scores, activation=None, name='dense_v2')

        with tf.variable_scope('sequence_tagging'):
            """
            sequence tagging
            """
            self.__build_sequence_tagging()

        with tf.variable_scope('optimizer'):
            """
            optimizer
            """
            self.__build_optimizer()

        """
        build session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        """
        build summary
        """
        self.__build_summary()

    def __create_feed_dict(self, batch, use_label=True):
        max_length_sentence = self.max_length_sentence
        max_length_word = self.max_length_word

        ip_char_ids, word_length = pad_char(sequences=[e['char_ids'] for e in batch], pad_tok=1, # 1 is padding
                                            max_length_word=max_length_word, max_length_sentence=max_length_sentence)
        ip_word_ids, sentence_length = pad_common(sequences=[e['word_ids'] for e in batch], pad_tok=1, # 1 is padding
                                                  max_length=max_length_sentence)

        res = {
            self.char_ids: ip_char_ids,
            self.word_ids: ip_word_ids,
            self.cap_ids: pad_common(sequences=[e['cap_ids'] for e in batch], pad_tok=0, max_length=max_length_sentence)[0], # 0 is padding
            self.pos_ids: pad_common(sequences=[e['pos_ids'] for e in batch], pad_tok=1, max_length=max_length_sentence)[0], # 1 is padding
            self.sentence_length: sentence_length,
            self.word_length: word_length,
        }

        if use_label:
            res[self.labels] = pad_common(sequences=[e['label_ids'] for e in batch], pad_tok=0, max_length=max_length_sentence)[0] # 0 is 'O' == 'Other'

        if self.id2reg != None:
            res[self.reg_ids] = pad_common(sequences=[e['reg_ids'] for e in batch], pad_tok=0, max_length=max_length_sentence)[0] # 0 is 'O' == 'Other'

        return res

    def batch_run(self, batch, i, mode='train',lr = None):
        ip_feed_dict = self.__create_feed_dict(batch)
        if lr != None: ip_feed_dict[self.lr_placeholder] = lr

        sentece_lengths = ip_feed_dict[self.sentence_length]

        if mode == 'train':
            ip_feed_dict[self.is_training] = True
            ip_feed_dict[self.dropout] = self.dropout_prob

            _, loss, summary = self.sess.run([self.train_op, self.train_loss, self.merged], feed_dict=ip_feed_dict)

            if i % self.freq_equal_summary == 0:
                self.train_writer.add_summary(summary, i)

            out = loss

        elif mode == 'test':
            ip_feed_dict[self.is_training] = False
            ip_feed_dict[self.dropout] = 1.0

            scores, transition, summary = self.sess.run( [self.fn_scores, self.transition, self.merged],
                                                        feed_dict=ip_feed_dict)
            predict_labels, _ = crf_decode_with_batch(scores=scores, sentence_lengths=sentece_lengths,
                                                      transition=transition)

            if i % self.freq_equal_summary == 0: self.test_writer.add_summary(summary, i)

            out = predict_labels

        return out

    def reset_graph(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def close_writer(self):
        self.train_writer.close()
        self.test_writer.close()

    def reset_dir_summary(self,dir_summary):
        self.dir_summary = dir_summary

        if tf.gfile.Exists(self.dir_summary):
            tf.gfile.DeleteRecursively(self.dir_summary)
        tf.gfile.MakeDirs(self.dir_summary)

        self.train_writer = tf.summary.FileWriter(self.dir_summary + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.dir_summary + '/test')


if __name__ == '__main__':
    pass
