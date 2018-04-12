from Components.component import Component
from Model.model import NERModel
from Components.utils import create_batch
import numpy as np
import os
import codecs
import tensorflow as tf

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

        if tf.gfile.Exists(self.dir_summary): tf.gfile.DeleteRecursively(self.dir_summary)
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

        """
        building parameters
        """
        dict = message['dictionary']
        dir_summary = os.path.join(config['saved_result_path'], 'Summary')
        f1_summary = F1Summary(dir_summary)

        parameters = {
            'id2char': dict['id2char'],
            'id2word': dict['id2word'],
            'id2label': dict['id2label'],
            'id2pos': dict['id2pos'],
            'id2cap': dict['id2cap'],
            'id2reg': dict['id2reg'],
            'dir_summary': dir_summary,
            'pre_emb_path': config['word2vec_path']
        }

        parameters.update(config['model_params'])

        """
        building model
        """
        self.model = NERModel(**parameters)
        self.model.build()

        """
        create batchs
        """
        batch_size = config['batch_size']
        train_dataset, test_dataset = message['dataset']['train'], message['dataset']['test']
        train_batch, test_batch = create_batch(dataset=train_dataset,batch_size=batch_size), \
                                  create_batch(dataset=test_dataset ,batch_size=batch_size)

        result_from_folds = {}

        for fold_id in range(max_fold):
            """
            reset model for each folds
            """
            self.model.reset_graph()
            self.model.reset_dir_summary(dir_summary + '/fold_%i/loss' % fold_id)
            f1_summary.reset(dir_summary + '/fold_%i/f1' % fold_id)

            print ('###########################')
            print ('### start training, with fold %i' % fold_id)
            print ('# all variables of model must be reinitialized ...')

            nepochs, freq_eval = config['epochs'], config['freq_eval']
            id2label, label2id = dict['id2label'], dict['label2id']
            train_i, eval_ti = 0, 0
            n_labels = len(id2label)

            init_lr = self.model.lr
            decay_lr_every = 50
            lr_decay = 0.9

            best_test = -np.inf

            for epoch in range(nepochs):
                for i, batch_id in enumerate(np.random.permutation(len(train_batch))):
                    batch = train_batch[batch_id]

                    loss = self.model.batch_run(batch=batch, i=train_i, mode='train',lr=init_lr)
                    train_i += 1

                    if train_i % decay_lr_every == 0:
                        init_lr *= lr_decay
                        print ('new lr: %f' % init_lr)

                    #print('-- fold %i, epoch %i, batch %i has loss %f' % (fold_id, epoch, i, loss))

                    if train_i % freq_eval == 0:
                        """
                        calculating score for dev/test set
                        """
                        print ('scoring for test ...')

                        predictions = []
                        conf_matrix = np.zeros((n_labels, n_labels), dtype=np.int32)

                        """
                        inference
                        """
                        for t_i, t_batch_id in enumerate(range(len(test_batch))):
                            t_batch = test_batch[t_batch_id]

                            batch_y_preds = self.model.batch_run(batch=t_batch, i=eval_ti, mode='test')
                            batch_r_preds = [elem['label_ids'] for elem in t_batch]

                            assert len(batch_y_preds) == len(batch_r_preds)
                            eval_ti += 1

                            batch_y_preds = [[id2label[i] for i in sample] for sample in batch_y_preds]
                            batch_r_preds = [[id2label[i] for i in sample] for sample in batch_r_preds]

                            """
                            predictions: a list of "<token> <expected_entity> <predict_entity>"
                            conf_matrix: a confusion matrix
                            --> both of their variables are used for calculate F1 (using CoNLL script), print result
                            """

                            for (data, y_preds, r_preds) in zip(t_batch, batch_y_preds, batch_r_preds):
                                for i, (y_pred, r_pred) in enumerate(zip(y_preds, r_preds)):
                                    new_line = " ".join([data['token'][i], r_preds[i], y_preds[i]])
                                    predictions.append(new_line)
                                    conf_matrix[label2id[r_pred], label2id[y_pred]] += 1
                                predictions.append("")

                        """
                        get result
                        """
                        test_score = self.display_eval_testset(predictions=predictions, conf_matrix=conf_matrix,
                                                               n_labels=n_labels, id2label=id2label,mode='test')

                        """
                        save result for visualizing using tensorboard
                        """
                        f1_summary.save(f1_age=test_score['f1_age'],f1_name=test_score['f1_name'],
                                        f1_total=test_score['f1_total'], i=eval_ti, mode='test')

                        """
                        save best result
                        """
                        if test_score['f1_total'] > best_test:
                            best_test = test_score['f1_total']

                            with open(test_score['path'], 'r') as f: best_result = f.read()
                            result_from_folds['fold_%d' % fold_id] = best_result

                            print(('-- New best score on test, ', str(best_test)))

        """
        close writer
        """
        for k,v in result_from_folds.items():
            path = os.path.join(config['saved_result_path'], "%s.txt" % k)
            with open(path,'w') as f: f.write(v)

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

        # confusion matrix with accuracy for each tag
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

        # global accuracy
        print("%i/%i (%.5f%%)" % (
            conf_matrix.trace(), conf_matrix.sum(), 100. * conf_matrix.trace() / max(1, conf_matrix.sum())
        ))

        return {
            'f1_age'  : float(eval_lines[2].strip().split()[-2]),
            'f1_name' : float(eval_lines[3].strip().split()[-2]),
            'f1_total': float(eval_lines[1].strip().split()[-1]),
            'path': scores_path
        }
