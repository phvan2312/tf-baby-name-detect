from Utils.component import Component

class Cap_component(Component):
    name = 'cap_extractor'
    provides = ['cap']
    requires = ['token']

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

        tokens = message.get('token')
        caps = []

        for _tokens in tokens:
            _caps = [self.__cap_feature(_token) for _token in _tokens]
            caps.append(_caps)

        message['cap']=caps

    def __cap_feature(self,s):
        """
        Capitalization feature:
        0 = low caps
        1 = all caps
        2 = first letter caps
        3 = one capital (not first letter)
        """
        if s.lower() == s:
            return 0
        elif s.upper() == s:
            return 1
        elif s[0].upper() == s[0]:
            return 2
        else:
            return 3

