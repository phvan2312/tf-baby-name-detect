from Utils.component import Component
from regmatcher import RegexMatcher

class Reg_component(Component):
    name = 'reg_extractor'
    provides = ['reg']
    requires = ['token']

    def __init__(self):
        Component.__init__(self)
        self.regex_matcher = RegexMatcher()
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
        regs = []

        for _tokens in tokens:
            _regs = self.regex_matcher.annotate_name(_tokens)
            regs.append(_regs)

        message['reg']=regs

