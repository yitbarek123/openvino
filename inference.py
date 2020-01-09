import os
import sys
import logging as log
#the model shout be downloaded first
from openvino.inference_engine import IENetwork,IECore

class Network:
    
    def __init__(self):
        self.plugin=None
        self.input_blob=None
        self.exec_network=None

    def load_model(self, model, device="CPU", cpu_extension=None):
        #load the model for the given IR files 
        model_xml=model
        model_bin=os.path.splittext(model_xml)[0]+".bin"
        # Initialize plugin
        self.plugin=IECore()
        # read the IR as a IENetwork
        network=IENetwork(model=model_xml, weights=model_bin)
        
        # Load the IENetwork into the plugin
        self.exec_network=self.plugin.load_network(network, device)
        
        #Get the input layer 
        self.input_blob=next(iter(network.inputs))
        
        #return the input shape(to dtermine preprocessing)
        return network.inputs[self.input_blob].shape
    
    def sync_inference(self, image):
        "make synchromous inference request, given an input image"
        self.exec_network.infer({self.input_blob:image})
        return
