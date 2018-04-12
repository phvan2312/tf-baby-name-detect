from Components.component import Component

class Cap_component(Component):
    name = 'cap_extractor'
    provides = ['ids.cap_ids','dictionary.id2cap']
    requires = ['data.sentence']

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

        tokens = message.get('data').get('sentence')
        caps = []

        for _tokens in tokens:
            _caps = [self.__cap_feature(_token) for _token in _tokens]
            caps.append(_caps)

        message['ids']['cap_ids']= caps
        message['dictionary']['id2cap'] = {i:str(i) for i in range(5)}

    def __cap_feature(self,s):
        """
        Capitalization feature:
        0 = padding
        1 = low caps
        2 = all caps
        3 = first letter caps
        4 = one capital (not first letter)
        """
        if s.lower() == s:
            return 1
        elif s.upper() == s:
            return 2
        elif s[0].upper() == s[0]:
            return 3
        else:
            return 4


