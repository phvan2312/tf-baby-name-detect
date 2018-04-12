from Components.component import Component
import os
import pandas as pd
from dateutil.parser import parse
from Components.utils import word_mapping,char_mapping,label_mapping

class Data_component(Component):
    name = 'data_extractor'
    provides = ['raw_data','dictionary','word_ids','char_ids','label_ids','pos_ids','token']
    requires = []

    def __init__(self):
        Component.__init__(self)
        pass

    # Load component from file
    def load(self, model_dir):
        Component.load(self,model_dir)
        pass

    # Store component
    def persist(self, model_dir):
        Component.load(self,model_dir)
        pass

    def lower_all(self,lst):
        return [[e.lower() for e in l] for l in lst]

    # Process message
    def process(self, message, config):

        Component.process(self,message,config)

        # Get absolute path & load file
        train_data_path = config.get('train_data_path')
        test_data_path  = config.get('test_data_path')
        word2vec_path = config.get('word2vec_path','')

        train_sents, train_labels, train_poss = self.load_file(train_data_path)
        print ('-- finished loading training dataset with %d samples' % len(train_sents))

        test_sents, test_labels, test_poss = self.load_file(test_data_path)
        print ('-- finished loading testing dataset with %d samples' % len(test_sents))

        sents, labels, poss = train_sents + test_sents, train_labels + test_labels, train_poss + test_poss

        # for save to mesasge
        raw_data = {}
        raw_data['sentence'] = sents
        raw_data['label'] = labels
        raw_data['pos'] = poss

        # We just be allowed to get information from training data only
        #train_data = {name:val[:split_id] for name,val in raw_data.items()}

        _, id2word, word2id    = word_mapping(lst_sentence=self.lower_all(train_sents),pre_emb=word2vec_path)
        _, id2char, char2id    = char_mapping(lst_sentence=self.lower_all(train_sents))
        _, id2label, label2id  = label_mapping(lst_x=train_labels, name='label')
        _, id2pos, pos2id      = word_mapping(lst_sentence=train_poss)

        word_ids  = []
        char_ids  = []
        label_ids = []
        pos_ids   = []

        for sent,label,pos in zip(raw_data['sentence'],raw_data['label'],raw_data['pos']):
            sent_lower = [word.lower() for word in sent]

            word_ids.append([word2id[w if w in word2id else '<unk>'] for w in sent_lower])
            char_ids.append([[char2id[c if c in char2id else '<unk>'] for c in w]
                     for w in sent_lower])
            label_ids.append([label2id[l] for l in label])
            pos_ids.append([pos2id[t if t in pos2id else '<unk>'] for t in pos])

        # for save to message
        dictionary = {
            'id2word': id2word,
            'id2char': id2char,
            'id2label': id2label,
            'id2pos': id2pos,
            'label2id' : label2id,
        }

        ids = {
            'word_ids' : word_ids,
            'char_ids' : char_ids,
            'label_ids': label_ids,
            'pos_ids'  : pos_ids
        }

        message['split_id'] = len(train_sents)
        message['data'] = raw_data
        message['dictionary'] = dictionary
        message['ids'] = ids

    # normalize token, convert from its variants to unique one
    def normalize_text(self,token):

        def valid_date(_token):
            try:
                parse(_token)
                return True
            except:
                return False

        token = token.strip()

        token = '<number>' if token.isdigit() else token
        token = '<punct>' if token in ['!', '?', ',', ':', ';'] else token
        token = '<date>' if valid_date(token) else token

        return token

    # load file from path
    # return list of sentences and its corresponding labels
    def load_file(self, path_to_file):
        assert os.path.isfile(path_to_file)

        df = pd.read_csv(path_to_file,encoding='utf-8')
        df.dropna(inplace=True)

        start_post, end_post = '[', ']'
        sents, labels, poss = [], [], []
        cur_sent, cur_label, cur_pos = [], [], []

        print('-- Start loading data from file --')

        for index, row in df.iterrows():
            token = self.normalize_text(row['token'])
            label = row['label']
            pos = row['pos']

            if token in [start_post, end_post]:
                if len(cur_sent) > 1:
                    sents.append(cur_sent)
                    labels.append(cur_label)
                    poss.append(cur_pos)
                cur_sent, cur_label, cur_pos = [], [], []
            else:
                cur_sent.append(token)
                cur_label.append(label)
                cur_pos.append(pos)

        print('-- Finished loading data --')
        print('-- Number of samples: ', len(sents))

        return (sents, labels, poss)