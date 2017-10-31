from Utils.component import Component
from Model.model import NERModel
from Utils.utils import create_batch
import numpy as np
import os
import codecs

class Entity_component(Component):
    name = 'entity_extractor'
    requires  = ['dictionary.id2char','dictionary.id2word','dictionary.id2label','dictionary.id2pos','dictionary.id2cap',
                'dictionary.id2reg','dataset.train','dataset.test']
    provides = ['trained_model']

    def __init__(self):
        Component.__init__(self)
        self.eval_temp = 'eval' #os.path.join(eval_path, "conlleval")
        self.eval_script = 'conlleval'
        pass

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

        parameters = {
            'id2char' : dict['id2char'],
            'id2word' : dict['id2word'],
            'id2label': dict['id2label'],
            'id2pos'  : dict['id2pos'],
            'id2cap'  : dict['id2cap'],
            'id2reg'  : dict['id2reg'],
            'char_emb_dim' : 20,
            'word_emb_dim' : 100,
            'cap_emb_dim'  : 10,
            'pos_emb_dim'  : 10,
            'reg_emb_dim'  : 5,
            'char_hid_dim' : 20,
            'word_hid_dim' : 100,
            'nn_for_char'  : 'bilstm',
            'dropout_prob' : .5,
            'lr' : .001,
            'optimize_method' : 'adam',
            'clip' : 5,
            'dir_summary' : 'summary'
        }
        # build model
        self.model = NERModel(**parameters)
        self.model.build()

        # create batch
        batch_size = config['batch_size']
        train_dataset, test_dataset = message['dataset']['train'], message['dataset']['test']
        train_batch, test_batch = create_batch(dataset=train_dataset,batch_size=batch_size), \
                                  create_batch(dataset=test_dataset ,batch_size=batch_size)

        # run
        print 'start training'
        nepochs = config['epochs']
        freq_eval = config['freq_eval']
        id2label = dict['id2label']
        label2id = dict['label2id']
        train_i = 0
        eval_i  = 0
        n_labels = len(id2label)

        for epoch in range(nepochs):
            print 'epoch %i starting ...' % epoch
            for i, batch_id in enumerate(np.random.permutation(len(train_batch))):
                batch = train_batch[batch_id]

                loss = self.model.batch_run(batch=batch,i=train_i,mode='train')
                train_i += 1

                print '-- batch %i has loss %f' % (i, loss)
                # caculate test set
                if i % freq_eval == 0:
                    predictions = []
                    count = np.zeros((n_labels,n_labels), dtype=np.int32)

                    for t_i, t_batch_id in enumerate(range(len(test_batch))):
                        t_batch = test_batch[t_batch_id]

                        batch_y_preds = self.model.batch_run(batch=t_batch, i=eval_i, mode='eval')
                        batch_r_preds = [elem['label_ids'] for elem in t_batch]

                        assert len(batch_y_preds) == len(batch_r_preds)
                        eval_i += 1

                        batch_y_preds = [[id2label[i] for i in sample] for sample in batch_y_preds]
                        batch_r_preds = [[id2label[i] for i in sample] for sample in batch_r_preds]

                        for (data,y_preds, r_preds) in zip(t_batch,batch_y_preds,batch_r_preds):
                            for i, (y_pred, r_pred) in enumerate(zip(y_preds, r_preds)):
                                new_line = ",".join([data['token'][i], r_preds[i], y_preds[i]])
                                predictions.append(new_line)
                                count[label2id[r_pred], label2id[y_pred]] += 1
                            predictions.append("")

                    eval_id = np.random.randint(1000000, 2000000)
                    output_path = os.path.join(self.eval_temp, "eval.%i.output" % eval_id)
                    scores_path = os.path.join(self.eval_temp, "eval.%i.scores" % eval_id)

                    with codecs.open(output_path, 'w', 'utf8') as f:
                        f.write("\n".join(predictions))
                    os.system("%s < %s > %s" % (self.eval_script, output_path, scores_path))

                    # CoNLL evaluation results
                    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
                    for line in eval_lines:
                        print line

                    # Confusion matrix with accuracy for each tag
                    print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_labels)).format(
                        "ID", "NE", "Total",
                        *([dict['id2label'][i] for i in xrange(n_labels)] + ["Percent"])
                    )
                    for i in xrange(n_labels):
                        print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_labels)).format(
                            str(i), dict['id2label'][i], str(count[i].sum()),
                            *([count[i][j] for j in xrange(n_labels)] +
                              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
                        )

                    # Global accuracy
                    print "%i/%i (%.5f%%)" % (
                        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
                    )

        self.model.close_writer()

    def __train(self):
        pass