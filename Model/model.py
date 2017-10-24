import tensorflow as tf
import numpy as np
from nn_utils import initialize_embedding,build_biRNN,build_fully_connected

class NERModel:
    def __init__(self,id2char,id2word,id2label,id2pos,id2cap,id2reg,
                 char_emb_dim,word_emb_dim,cap_emb_dim,pos_emb_dim,reg_emb_dim,
                 char_hid_dim,word_hid_dim,
                 nn_for_char,dropout_prob):
        self.id2char = id2char
        self.id2word = id2word
        self.id2label = id2label
        self.id2pos = id2pos
        self.id2cap = id2cap
        self.id2reg = id2reg

        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.cap_emb_dim = cap_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.reg_emb_dim = reg_emb_dim

        self.char_hid_dim = char_hid_dim
        self.word_hid_dim = word_hid_dim

        self.nn_for_char = nn_for_char
        self.dropout_prob = dropout_prob

    def __build_placeholder(self):
        self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None],
                                       name='char_ids') # batch x max_len_sent x max_len_word
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids') # batch x max_len_sent
        self.cap_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='cap_ids') # batch x max_len_sent
        self.pos_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos_ids') # batch x max_len_sent
        self.reg_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='reg_ids') # batch x mat_len_sent

        self.sentence_length = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_length') # bacth
        self.word_length = tf.placeholder(dtype=tf.int32,shape=[None,None],name='word_length') # batch x max_len_sent

    def __build_word_input(self):
        #
        # Initialize some embedding matrix
        #
        self.CHAR_EMB = initialize_embedding(shape=(len(self.id2char), self.char_emb_dim), name='character')
        self.WORD_EMB = initialize_embedding(shape=(len(self.id2word), self.word_emb_dim), name='word')
        self.CAP_EMB = initialize_embedding(shape=(len(self.id2cap), self.cap_emb_dim), name='cap')
        self.POS_EMB = initialize_embedding(shape=(len(self.id2pos), self.pos_emb_dim), name='pos')
        self.REG_EMB = initialize_embedding(shape=(len(self.id2reg), self.reg_emb_dim), name='reg')

        self.char_enc = tf.nn.embedding_lookup(self.CHAR_EMB, self.char_ids)
        self.word_enc = tf.nn.embedding_lookup(self.WORD_EMB, self.word_ids)
        self.cap_enc = tf.nn.embedding_lookup(self.CAP_EMB, self.cap_ids)
        self.pos_enc = tf.nn.embedding_lookup(self.POS_EMB, self.pos_ids)
        self.reg_enc = tf.nn.embedding_lookup(self.REG_EMB, self.reg_ids)

        #
        # Build word representation by concat all feature representation
        #
        word_from_char = self.__build_word_from_char()
        self.fn_word_enc = tf.concat([self.word_enc,word_from_char,self.cap_enc,self.pos_enc,self.reg_enc],axis=2)


    # Build word representation from character level
    # We have two approach:
    # First, using Bi-LSTM for Char (default)
    # Second, using CNN
    def __build_word_from_char(self):
        def char_bilstm():
            return build_biRNN(input=self.char_enc,hid_dim=self.char_hid_dim,sequence_length=self.word_length,
                                      cells=None,mode='character')

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

    def process(self,input,is_train):
        #
        # initialize model and build word representation
        #
        self.__build_placeholder()
        self.__build_word_input()

        #
        # apply two bi-LSTM for word representation, then dropout some units
        #
        self.fn_w_v1 = build_biRNN(input=self.fn_word_enc,hid_dim=self.word_hid_dim,sequence_length=self.sentence_length,
                              cells=None,mode='other')

        self.fn_w_v2 = build_biRNN(input=self.fn_w_v1,hid_dim=self.word_hid_dim,sequence_length=self.sentence_length,
                                   cells=None,mode='other')

        self.fn_output = tf.nn.dropout(self.fn_w_v2,keep_prob=self.dropout_prob)

        #
        # fully connected layer to project from (2 x word_hid_dim) shape to (n_tag)
        #
        num_label = len(self.id2label)
        tanh_scores = build_fully_connected(in_dim=2*self.word_hid_dim,out_dim=num_label,
                                            input=self.fn_output,activation='tanh')
        self.fn_scores = build_fully_connected(in_dim=num_label,out_dim=num_label,input=tanh_scores,activation=None)

        #
        # Sequence level tagging (or CRF)
        #
        raise NotImplemented()





