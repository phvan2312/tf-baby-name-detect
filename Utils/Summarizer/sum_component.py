from Utils.component import Component
import os

class Sum_component(Component):
    name = 'sum_extractor'
    provides = ['dataset']
    requires = ['data.sentence', 'ids.word_ids', 'ids.char_ids', 'ids.label_ids', 'ids.pos_ids', 'ids.cap_ids']

    def __init__(self):
        Component.__init__(self)
        pass

    # Load component from file
    def load(self, model_dir):
        Component.load(self, model_dir)
        pass

    # Store component
    def persist(self, model_dir):
        Component.load(self, model_dir)
        pass

    # Process message
    def process(self, message, config):
        Component.process(self, message, config)

        dataset = []
        ids = message['ids']

        for token, word_ids, char_ids, label_ids, pos_ids, cap, reg in zip(message['data']['sentence'],ids['word_ids'],
                                                                    ids['char_ids'], ids['label_ids'], ids['pos_ids'],
                                                                    ids['cap_ids'], ids['reg_ids']):
            data = {}
            data['token'] = token
            data['word_ids']  = word_ids
            data['char_ids']  = char_ids
            data['label_ids'] = label_ids
            data['pos_ids']   = pos_ids
            data['cap_ids']   = cap
            data['reg_ids']   = reg

            dataset.append(data)

        split_id = message['split_id']
        train_data = dataset[:split_id]
        test_data = dataset[split_id:]

        message['dataset'] = {'train': train_data, 'test': test_data}

