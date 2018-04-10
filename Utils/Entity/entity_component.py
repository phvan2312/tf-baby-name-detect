from Utils.component import Component
from Model.model import NERModel
from Utils.utils import create_batch
import numpy as np
import os
import codecs
import tensorflow as tf

dir_summary = 'Summary'
max_fold = 3

class F1Summary:
    def __init__(self, dir_summary):
        self.dir_summary = dir_summary
        if tf.gfile.Exists(self.dir_summary):
            tf.gfile.DeleteRecursively(self.dir_summary)
        tf.gfile.MakeDirs(self.dir_summary)

        self.init()

    def init(self):
        with tf.Graph().as_default():
            # build placeholder
            self.f1_name = tf.placeholder(dtype=tf.float32,shape=[],name='f1')
            self.f1_age = tf.placeholder(dtype=tf.float32, shape=[], name='f1')
            self.f1_total = tf.placeholder(dtype=tf.float32, shape=[], name='f1')

            # build summary
            tf.summary.scalar('f1_name',self.f1_name)
            tf.summary.scalar('f1_age', self.f1_age)
            tf.summary.scalar('f1_total', self.f1_total)

            self.merged = tf.summary.merge_all()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            self.train_writer = tf.summary.FileWriter(self.dir_summary + '/train')
            self.test_writer = tf.summary.FileWriter(self.dir_summary + '/test')

    def reset(self,dir_summary):
        self.dir_summary = dir_summary

        if tf.gfile.Exists(self.dir_summary):
            tf.gfile.DeleteRecursively(self.dir_summary)
        tf.gfile.MakeDirs(self.dir_summary)

        self.train_writer = tf.summary.FileWriter(self.dir_summary + '/train')
        self.test_writer = tf.summary.FileWriter(self.dir_summary + '/test')

    def save(self,f1_name,f1_age,f1_total,i,mode):
        if mode == 'train':
            writer = self.train_writer
        elif mode == 'test':
            writer = self.test_writer

        summary = self.sess.run(self.merged,feed_dict={
            self.f1_name : f1_name,
            self.f1_age  : f1_age,
            self.f1_total: f1_total
        })

        writer.add_summary(summary,i)

class Entity_component(Component):
    name = 'entity_extractor'
    requires  = ['dictionary.id2char','dictionary.id2word','dictionary.id2label','dictionary.id2pos','dictionary.id2cap',
                'dictionary.id2reg','dataset.train','dataset.test']
    provides = ['trained_model']

    def __init__(self):
        Component.__init__(self)
        eval_path = os.path.dirname(os.path.realpath(__file__))

        self.eval_dir   = os.path.join(eval_path,'Evaluation')
        self.eval_script = os.path.join(eval_path,'conlleval')

    # Load component from file
    def load(self, model_dir):
        Component.load(self,model_dir)
        pass

    # Store component
    def persist(self, model_dir):
        Component.load(self,model_dir)
        pass

    # Process message
    def process(self, message, config):
        Component.process(self, message, config)
        dict = message['dictionary']
        f1_summary = F1Summary(dir_summary)

        parameters = {
            'id2char' : dict['id2char'],
            'id2word' : dict['id2word'],
            'id2label': dict['id2label'],
            'id2pos'  : dict['id2pos'],
            'id2cap'  : dict['id2cap'],
            'id2reg'  : dict['id2reg'],
            'char_emb_dim' : 25, # 25 for bi-lstm is ok
            'word_emb_dim' : 100, # must be match with pretrained word2vec
            'cap_emb_dim'  : 10,
            'pos_emb_dim'  : 10,
            'reg_emb_dim'  : 5,
            'char_hid_dim' : 20,
            'word_hid_dim' : 50,
            'nn_for_char'  : 'cnn', # must be 'bilstm' or 'cnn'
            'filter_sizes' : [2,3,4,5,6],
            'num_filter'   : 20,
            'dropout_prob' : 0.5,
            'lr' : .002,
            'optimize_method' : 'adam',
            'clip' : 1,
            'dir_summary' : dir_summary,
            'pre_emb_path': config['word2vec_path'],
            'max_length_word': 20,
            'max_length_sentence': 100
        }
        # build model
        self.model = NERModel(**parameters)
        self.model.build()

        # create batch
        batch_size = config['batch_size']
        train_dataset, test_dataset = message['dataset']['train'], message['dataset']['test']
        train_batch, test_batch = create_batch(dataset=train_dataset,batch_size=batch_size), \
                                  create_batch(dataset=test_dataset ,batch_size=batch_size)

        result_from_folds = []

        for fold_id in range(max_fold):
            # need to reset model for each fold
            # reset model, thay doi diem save
            self.model.reset_graph()
            self.model.reset_dir_summary(dir_summary + '/fold_%i/loss' % fold_id)
            f1_summary.reset(dir_summary + '/fold_%i/f1' % fold_id)

            print ('##########################')
            print ('### start training, with fold %i' % fold_id)
            print ('# all variables of model must be reinitialized ...')

            nepochs, freq_eval = config['epochs'], config['freq_eval']
            id2label, label2id = dict['id2label'], dict['label2id']
            train_i, eval_ti = 0, 0
            n_labels = len(id2label)

            best_test = -np.inf

            for epoch in range(nepochs):
                for i, batch_id in enumerate(np.random.permutation(len(train_batch))):
                    batch = train_batch[batch_id]

                    loss = self.model.batch_run(batch=batch, i=train_i, mode='train')
                    train_i += 1

                    print('-- fold %i, epoch %i, batch %i has loss %f' % (fold_id, epoch, i, loss))
                    # caculate test/dev set
                    if i % freq_eval == 0:
                        print ('scoring for test ...')

                        ########################
                        ########### FOR TEST SET
                        predictions = []
                        count = np.zeros((n_labels, n_labels), dtype=np.int32)

                        for t_i, t_batch_id in enumerate(range(len(test_batch))):
                            t_batch = test_batch[t_batch_id]

                            batch_y_preds = self.model.batch_run(batch=t_batch, i=eval_ti, mode='test')
                            batch_r_preds = [elem['label_ids'] for elem in t_batch]

                            assert len(batch_y_preds) == len(batch_r_preds)
                            eval_ti += 1

                            batch_y_preds = [[id2label[i] for i in sample] for sample in batch_y_preds]
                            batch_r_preds = [[id2label[i] for i in sample] for sample in batch_r_preds]

                            for (data, y_preds, r_preds) in zip(t_batch, batch_y_preds, batch_r_preds):
                                for i, (y_pred, r_pred) in enumerate(zip(y_preds, r_preds)):
                                    new_line = " ".join([data['token'][i], r_preds[i], y_preds[i]])
                                    predictions.append(new_line)
                                    count[label2id[r_pred], label2id[y_pred]] += 1
                                predictions.append("")

                        # display result
                        test_score = self.display_eval_testset(predictions=predictions, conf_matrix=count,
                                                               n_labels=n_labels, id2label=id2label,mode='test')
                        # save f1 summary
                        f1_summary.save(f1_age=test_score['f1_age'],f1_name=test_score['f1_name'],
                                        f1_total=test_score['f1_total'], i=eval_ti, mode='test')

                        # save best score
                        if test_score['f1_total'] > best_test:
                            best_test = test_score['f1_total']
                            print(('-- New best score on test, ', str(best_test)))

            result_from_folds.append({'best_test':best_test})
        self.model.close_writer()

    def display_eval_testset(self,predictions,conf_matrix,n_labels,id2label,mode):
        eval_id = np.random.randint(1000000, 2000000)
        output_path = os.path.join(self.eval_dir, "eval_%s.%i.output" % (mode,eval_id))
        scores_path = os.path.join(self.eval_dir, "eval_%s.%i.scores" % (mode,eval_id))

        with codecs.open(output_path, 'w', 'utf8') as f:
            f.write("\n".join(predictions))
        os.system("%s < %s > %s" % (self.eval_script, output_path, scores_path))

        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
        for line in eval_lines:
            print(line)

        # Confusion matrix with accuracy for each tag
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_labels)).format(
            "ID", "NE", "Total",
            *([id2label[i] for i in range(n_labels)] + ["Percent"])
        ))
        for i in range(n_labels):
            print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_labels)).format(
                str(i), id2label[i], str(conf_matrix[i].sum()),
                *([conf_matrix[i][j] for j in range(n_labels)] +
                  ["%.3f" % (conf_matrix[i][i] * 100. / max(1, conf_matrix[i].sum()))])
            ))

        # Global accuracy
        print("%i/%i (%.5f%%)" % (
            conf_matrix.trace(), conf_matrix.sum(), 100. * conf_matrix.trace() / max(1, conf_matrix.sum())
        ))
        
        #print (eval_lines)
        return {
            'f1_age'  : float(eval_lines[2].strip().split()[-2]),
            'f1_name' : float(eval_lines[3].strip().split()[-2]),
            'f1_total': float(eval_lines[1].strip().split()[-1]),
        }
