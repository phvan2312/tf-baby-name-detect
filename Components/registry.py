import os
from .Cap.cap_component import Cap_component
from .Regex.reg_component import Reg_component
from .Data.data_component import Data_component
from .Summarizer.sum_component import Sum_component
from .Entity.entity_component import Entity_component
import collections

class Registry:
    component_classes = [Cap_component,Reg_component,Data_component,Sum_component,Entity_component]
    registered_components = {c.name:c for c in component_classes}

    def __init__(self):
        pass

    def pipeline_to_components(self,pipeline):
        assert type(pipeline) is list
        # check all components used in pipeline have already been registered
        for component in pipeline:
            assert component in self.registered_components

        components = collections.OrderedDict()
        for component in pipeline:
            components[component] = self.registered_components[component]()

        return components

if __name__ == '__main__':
    pass