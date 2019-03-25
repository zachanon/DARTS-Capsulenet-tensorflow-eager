import numpy as np
import tensorflow as tf
import operations as ops

from capsules import CapsuleLayer, Length, squash

class ConvolutionalBlock(tf.keras.Model):
    
    def __init__(self, N=5, channels=128):
        super(ConvolutionalBlock, self).__init__()
        
        graph = []
        stdconv = []

        for i in range(N):
            graph.append([])
            stdconv.append(tf.keras.layers.SeparableConv2D(filters=ops.NODE_CHANNELS, kernel_size=1))
            for j in range(i):
                #the operation from node j to node i
                graph[i].append(ops.MixedOperation())
                
        self.graph = graph
        self.stdconv = stdconv
        self.outconv = tf.keras.layers.SeparableConv2D(filters=channels, kernel_size=1)

    def call(self, x, a):
        
        #unpack variables
        #graph = self.graph
        N = len(self.graph)
        
        #first node is the input
        nodes = [x]
        
        #for every node in the graph
        for i in range(N):
            
            #we already have the first node
            if i>0:
                nodes.append([])
            
            #for every edge j leading to node i
            for j in range(i):
                
                #add the value of the operation on edge i,j to node i, weighted by architecture[i,j]
                nodes[i].append(self.graph[i][j](nodes[j], a[:,i,j]))
                
            #concat the incoming channels
            nodes[i] = tf.concat(nodes[i], axis=3)
            
            #reduce number of filters
            nodes[i] = self.stdconv[i](nodes[i])
            
        #concat the outgoing block
        nodes = tf.concat(nodes[i], axis=3)

        #reduce number of filters
        nodes = self.outconv(nodes)   
        
        return nodes
    
class CapsuleBlock(tf.keras.Model):
    
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
        
    
class DensePredictor(tf.keras.Model):
    
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
    
class ConvModel(tf.keras.Model):
    
    def __init__(self, N=5, hidden=128, num_class=10):
        super(ConvModel, self).__init__()
        self.conv = ConvolutionalBlock(N=N)
        self.pred = DensePredictor(hidden=hidden, num_class=num_class)

    def call(self, x, a):
        
        conv = self.conv(x, a)
        pred = self.pred(conv)
        
        return pred
    
class CapsuleNet(tf.keras.Model):
    
    def __init__(self, dim_capsule, num_class=10):
        super(CapsuleNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=256, kernel_size=9, activation=tf.nn.relu)
        self.prim = tf.keras.layers.Conv2D(filters=8*32, kernel_size=9, strides=2)
        self.reshape = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule])
        self.squash = tf.keras.layers.Lambda(squash)
        self.caps = CapsuleLayer(num_class, dim_capsule, routings=3)
        self.pred = Length()
        
    def call(self, x):
        
        conv = self.conv(x)
        prim = self.prim(conv)
        reshape = self.reshape(prim)
        squash = self.squash(reshape)
        caps = self.caps(squash)
        pred = self.pred(caps)
        
        return pred
    
class CapsuleModel(tf.keras.Model):
    def __init__(self, N=5, dim_capsule=16, num_class=10):
        super(CapsuleModel, self).__init__()
        self.conv = ConvolutionalBlock(N=N)
        self.caps = CapsuleBlock(dim_capsule=dim_capsule, num_capsule=num_class)
        self.pred = Length()
        
    def call(self, x, a):
        
        conv = self.conv(x, a)
        caps = self.caps(conv)
        pred = self.pred(caps)
        
        return pred
        