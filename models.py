import numpy as np
import tensorflow as tf
import operations as ops

from capsules import CapsuleLayer, Length, squash

class ConvolutionalBlock(tf.keras.Model):
    """ A block making use of the DARTS algorithm to build an optimized convolutional architecture for different use-cases
            - N: the number of nodes in the block
        Returns a tensor of shape [batch, H, W, (channels_in + N*NODE_CHANNELS)] where H, W are defined by the input,
        and NODE_CHANNELS is a hyperparameter.
        """
    def __init__(self, N=5):
        super(ConvolutionalBlock, self).__init__()
        
        graph = []
        
        #build a directed acyclic graph feauting mixed operations on the edges
        for i in range(N):
            graph.append([])
            for j in range(i):
                #the operation from node j to node i
                graph[i].append(ops.MixedOperation())
                
        self.graph = graph

    def call(self, x, a):
        
        #unpack variables
        #graph = self.graph
        N = len(self.graph)
        
        #first node is the input
        nodes = [x]
        
        #for every node in the graph
        for i in range(N):
            
            if i>0:
                nodes.append([])
            #we already have the first node
            
            #for every edge j leading to node i
            for j in range(i):
                
                #add the value of the operation on edge i,j to node i, weighted by architecture[i,j]
                nodes[i].append(self.graph[i][j](nodes[j], a[:,i,j]))
                
            if i>0:
                nodes[i] = tf.add_n(nodes[i])
                
        #final value is concat of the nodes
        nodes = tf.concat(nodes, axis=3)  
        
        return nodes
    
class DensePredictor(tf.keras.Model):
    """ Basic two-layer fully-connected classifier:
            - hidden: number of units in the hidden layer
            - num_class: number of potential classifications
        Returns a tensor of [batch, num_class] logits
        """
    
    def __init__(self, hidden, num_class):
        super(DensePredictor, self).__init__()
        self.flat = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=hidden, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=num_class)
        
    def call(self, inputs):
        
        flat = self.flat(inputs)
        hidden = self.hidden(flat)
        preds = self.out(hidden)
        
        return preds
    
class ConvolutionalModel(tf.keras.Model):
    """ A CNN making use of the DARTS algorithm and a fully-connected classifier. Utilizes only one cell, no max pooling
            - N: the number of nodes in the DARTS cell
            - channels: the number of channels in the block given to the fully connected predictor
            - hidden: the number of units in the hidden layer of the predictor
            - num_class: the number of potential classification targets
        Returns a tensor of [batch, num_class] logits. Requires an architecture when calling
        """
    
    def __init__(self, N=5, channels=128, hidden=128, num_class=10):
        super(ConvModel, self).__init__()
        self.conv = ConvolutionalBlock(N=N)
        self.outconv = tf.keras.layers.SeparableConv2D(filters=channels, kernel_size=1)
        self.pred = DensePredictor(hidden=hidden, num_class=num_class)

    def call(self, x, a):
        
        conv = self.conv(x, a)
        outconv = self.outconv(conv)
        pred = self.pred(conv)
        
        return pred
    
class CapsuleBlock(tf.keras.Model):
    """ An ease-of-use block featuring the dynamic routing algorithm. Input gets reshaped into capsules and routed.
            - dim_capsule: length of the output vectors (capsules)
            - num_capsule: the number of capsules for the input to be sorted into
            - routings: the number of passes through the dynamic routing algorithm
        Returns a tensor of shape [batch, dim_capsule, num_capsule]
        """
    
    def __init__(self, dim_capsule, num_capsule, routings=3):
        super(CapsuleBlock, self).__init__()
        self.reshape = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule])
        self.squash = tf.keras.layers.Lambda(squash)
        self.routing = CapsuleLayer(num_capsule, dim_capsule, routings=routings)
        
    def call(self, inputs):
        capsules = self.reshape(inputs)
        squashed = self.squash(capsules)
        routed = self.routing(squashed)
        return routed
    
class CapsuleModel(tf.keras.Model):
    """A Capsule Net making use of the DARTS algorithm and a dynamic routing classifier. Utilizes only one cell, no max pooling
            - dim_capsule, num_class, routings: hyperparameters for the capsule block
            - N: the number of nodes in the DARTS cell
            - convolve_cnn_block: a binary modifier determining whether to apply a convolution to the concatenated nodes of the 
                convolutional block
            - channels, kernel_size, strides, dilation_rate: hyperparameters for the optional convolutional layer between 
                the convolutional block and the capsule block
        Returns a tensor of [batch, dim_capsule, num_class] prediction vectors. Requires an architecture when calling
        """
    
    def __init__(self, dim_capsule=16, num_class=10, routings=3,
                 N=5,
                 convolve_cnn_block=False,
                 channels=128, kernel_size=1, strides=1, dilation_rate=1):
        
        super(CapsuleModel, self).__init__()
        self.convolve_cnn_block = convolve_cnn_block
        
        self.conv = ConvolutionalBlock(N=N)
        if convolve_cnn_block:
            self.cnnconv = tf.keras.layers.SeparableConv2D(filters=channels, kernel_size=kernel_size,
                                                      strides=strides, dilation_rate=dilation_rate)
        self.caps = CapsuleBlock(dim_capsule=dim_capsule, num_capsule=num_class, routings=routings)
        self.pred = Length()
        
    def call(self, x, a, *args):
        
        conv = self.conv(x, a)
        if self.convolve_cnn_block:
            conv = self.cnnconv(conv)
        caps = self.caps(conv)
        pred = self.pred(caps)
        
        return pred