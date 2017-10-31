import numpy
import os

class Component:
    # Name of the component to be used when integrating it in a pipeline.
    name = ""
    # Defines what attributes the pipeline component will provide when called.
    provides = []
    # Which attributes on a message are required by this component.
    requires = []

    def __init__(self):
        self.split_token = '.'
        pass

    # Load component from file
    def load(self,model_dir):
        pass

    # Store component
    def persist(self,model_dir):
        pass

    # Process message
    def process(self,message,config):
        # check required attributes in message
        print '-- %s start processing.' % self.name

        assert type(message) is dict
        assert type(config) is dict

        for require in self.requires:
            if self.split_token in require:
                reqs = require.split(self.split_token)
            else:
                reqs = [require]
            self.__validate(reqs,message)

    def __validate(self,reqs,data):
        for req in reqs:
            assert req in data
            data = data.get(req)


if __name__ == '__main__':
    component = Component()
    component.process({},{})
