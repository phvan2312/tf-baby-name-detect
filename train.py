import tensorflow as tf
import os, json, datetime, shutil
from Components.registry import Registry

flags = tf.app.flags

flags.DEFINE_string('word2vec_path','./Data/Word2vec/43k_word2vec.bin','path for storing word representation (.bin only)')
flags.DEFINE_string('train_data_path','./Data/Train/fold_0/train.csv','path for training phase')
flags.DEFINE_string('test_data_path','./Data/Train/fold_0/test.csv','path for testing phase')
flags.DEFINE_integer('epochs',30,'number of epochs')
flags.DEFINE_integer('freq_eval',20,'number of batch passed to evaluate test set')
flags.DEFINE_string('model_params_path','./model_params.json','model parameters path')
flags.DEFINE_integer('batch_size',25,'number of samples per batch')
flags.DEFINE_string('saved_result_path','./Results','folder for saving result')

FLAGS = tf.app.flags.FLAGS
pipeline = ['data_extractor','cap_extractor','reg_extractor','sum_extractor','entity_extractor']

"""
build config for pipeline
"""
def build_config():
    config = {}

    config['word2vec_path'] = FLAGS.word2vec_path if FLAGS.word2vec_path != '' else ''
    if config['word2vec_path'] != '': assert os.path.isfile(config['word2vec_path'])

    config['train_data_path'] = FLAGS.train_data_path
    assert os.path.isfile(config['train_data_path'])

    config['test_data_path'] = FLAGS.test_data_path
    assert os.path.isfile(config['test_data_path'])

    config['epochs'] = FLAGS.epochs
    assert config['epochs'] > 0

    config['freq_eval'] = FLAGS.freq_eval
    assert config['freq_eval'] > 0

    config['batch_size'] = FLAGS.batch_size
    assert config['batch_size'] > 0

    assert os.path.isfile(FLAGS.model_params_path)
    config['model_params'] = json.load(open(FLAGS.model_params_path,'r'))

    assert os.path.isdir(FLAGS.saved_result_path)
    config['saved_result_path'] = FLAGS.saved_result_path #os.path.join(FLAGS.saved_result_path, str(datetime.datetime.now()).replace(' ','_') )

    """
    save some necessary materials in new folder.
    """
    json.dump(config['model_params'], open(os.path.join(config['saved_result_path'],'model_params.json'),'w'))

    return config

def main(_):
    registry = Registry()
    config = build_config()

    print json.dumps(config, indent=2, sort_keys=True)

    components = registry.pipeline_to_components(pipeline)
    message = {}

    for name,component in components.items():
        component.process(message=message,config=config)

if __name__ == '__main__':
    tf.app.run()
