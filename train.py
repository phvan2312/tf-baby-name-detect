import tensorflow as tf
import numpy as np
import os
from Utils.registry import Registry
from Model.model import NERModel

flags = tf.app.flags

flags.DEFINE_string('word2vec_path','./Data/Word2vec/43k_word2vec.bin','path for storing word representation (.bin only)')
flags.DEFINE_string('data_path','./Data/Train/train.csv','path for storing data')
flags.DEFINE_float('test_size',.2,'% of test size')
flags.DEFINE_integer('epochs',12,'number of epochs')
flags.DEFINE_integer('freq_eval',500,'number of samples passed to evaluate test set')
flags.DEFINE_boolean('use_regex',True,'use regular expression or not')
flags.DEFINE_integer('batch_size',32,'number of samples per batch')

FLAGS = tf.app.flags.FLAGS
pipeline = ['data_extractor','cap_extractor','reg_extractor','sum_extractor','entity_extractor']

# build config for pipeline
def build_config():
    config = {}

    config['word2vec_path'] = FLAGS.word2vec_path if FLAGS.word2vec_path != '' else ''
    if config['word2vec_path'] != '':
        print ('-- Word2vec path: ', config['word2vec_path'])
    else:
        print ('-- Word2vec is not used !!')

    config['data_path'] = FLAGS.data_path if os.path.exists(FLAGS.data_path) else -1
    if config['data_path'] == -1:
        raise Exception('Data path does not exists !!')
    else:
        print ('-- Data path: ', config['data_path'])

    config['test_size'] = FLAGS.test_size
    print ('-- Test data proportion: ', config['test_size'])

    config['epochs'] = FLAGS.epochs
    print ('-- Number of epochs: ', config['epochs'])

    config['freq_eval'] = FLAGS.freq_eval
    print ('-- Number of samples passed to evaluate test data: ', config['freq_eval'])

    config['use_regex'] = FLAGS.use_regex
    print ('-- Using regular expression as an addtional feature: ', config['use_regex'])

    config['batch_size'] = FLAGS.batch_size
    print ('-- Number of samples per batch: ', config['batch_size'])

    return config

def main(_):
    registry = Registry()
    config = build_config()

    components = registry.pipeline_to_components(pipeline)
    message = {}

    for name,component in components.iteritems():
        component.process(message=message,config=config)

    print message

    '''
    def __init__(self, id2char, id2word, id2label, id2pos, id2cap, id2reg,
                 char_emb_dim, word_emb_dim, cap_emb_dim, pos_emb_dim, reg_emb_dim,
                 char_hid_dim, word_hid_dim,
                 nn_for_char, dropout_prob, lr, optimize_method, clip,
                 dir_summary):
    '''

if __name__ == '__main__':
    tf.app.run()