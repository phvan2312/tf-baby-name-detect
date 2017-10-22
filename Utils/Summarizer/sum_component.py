from Utils.component import Component
import os
from Utils.utils import pad_word_chars

class Sum_component(Component):
    name = 'sum_extractor'
    provides = ['dataset']
    requires = ['token','word_ids','char_ids','label_ids','pos_ids','cap']

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

    # Process message
    def process(self, message, config):
        Component.process(self,message,config)

        use_regex = config.get('use_regex',False)
        if use_regex:
            message['reg'] = [-1] * len(message['token'])

        dataset = []
        for token,word_ids,char_ids,label_ids,pos_ids,cap,reg in zip(message['token'],message['word_ids'],
                        message['char_ids'],message['label_ids'],message['pos_ids'],message['cap'],message['reg']):
            data = {}

            data['token'] = token
            data['word_ids'] = word_ids
            data['char_ids'] = pad_word_chars(char_ids,max_word_len=config.get('max_word_len',-1))
            data['label_ids'] = label_ids
            data['pos_ids'] = pos_ids
            data['cap'] = cap
            data['reg'] = reg

            dataset.append(data)

        test_size = config.get('test_size',.2)
        split_id = int(len(dataset) * (1 - test_size))
        train_data = dataset[:split_id]
        test_data = dataset[split_id:]

        message['dataset'] = {'train':train_data,'test':test_data}


