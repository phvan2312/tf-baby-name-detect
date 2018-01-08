from Utils.component import Component
from .regmatcher import RegexMatcher
from Utils.utils import common_mapping

class Reg_component(Component):
    name = 'reg_extractor'
    provides = ['ids.reg_ids','dictionary.id2reg']
    requires = ['data.sentence']

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

        tokens = message.get('data').get('sentence')
        regs = []

        for _tokens in tokens:
            _regs = self.regex_matcher.annotate_name(_tokens)
            regs.append(_regs)

        _,id2reg,_ = common_mapping(regs,'regex')

        message['ids']['reg_ids']=regs
        message['dictionary']['id2reg'] = id2reg

