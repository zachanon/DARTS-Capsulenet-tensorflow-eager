import numpy as np
import tensorflow as tf


NODE_CHANNELS = 64

class Zero(tf.keras.Model):
    """Uses tf.zeros_like method to return a zeros tensor with the same shape and dtype as input.
    If variable does not have enough channels, tiles the input
    """
    def __init__(self):
        super(Zero, self).__init__()
        
    def call(self, x):
        
        if x.shape[3] < NODE_CHANNELS:
            tiled = tf.tile(x, multiples=(1,1,1,128))
            zero = tf.zeros_like(tiled)
            return tiled
        
        zero = tf.zeros_like(x)
        return zero
    
class Identity(tf.keras.Model):
    """Identity block, simply returns the input variable
    If variable does not have enough channels, tiles the input"""
    
    def __init__(self):
        super(Identity, self).__init__()
        
    def call(self, x):
        
        if x.shape[3] < NODE_CHANNELS:
            tiled = tf.tile(x, multiples=(1,1,1,128))
            return tiled
        
        return x
    
class SeparableConvolution3x3(tf.keras.Model):
    """Applies relu activation function, 3x3 separable convolution, and batchnormalizes the output"""
    
    def __init__(self):
        super(SeparableConvolution3x3, self).__init__()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.SeparableConv2D(filters=NODE_CHANNELS,
                                                    kernel_size=3,)
        self.bn = tf.keras.layers.BatchNormalization()
        
    def call(self, x):
        relu = self.relu(x)
        conv = self.conv(relu)
        norm = self.bn(conv)
        
        #pad with zeros to maintain input shape
        hd, wd = int(x.shape[1]-norm.shape[1]), int(x.shape[2]-norm.shape[2])
        padding = int(hd/2), int(wd/2)
        
        padded = tf.keras.layers.ZeroPadding2D(padding=padding)(norm)
        
        return padded
    
class StandardizingConvolution(tf.keras.Model):
    """Reduces channel depth using a 1x1 convolution"""
    
    def __init__(self, filters=NODE_CHANNELS):
        super(StandardizingConvolution, self).__init__()
        self.conv = tf.keras.layers.SeparableConv2D(filters=filters,
                                    kernel_size=1)
        
    def call(self, x):
        out = self.conv(x)
        return out
    
OPERATIONS = {
    'identity': Identity,
    'sep_conv_3x3': SeparableConvolution3x3,
    'none': Zero,
}

CONV_OPS = {
    'identity': Identity,
    'sep_conv_3x3': SeparableConvolution3x3,
    'none': Zero,
}
    
class MixedOperation(tf.keras.Model):
    """Mixed operation for training architecture"""
    
    def __init__(self):
        super(MixedOperation, self).__init__()
        operations = []
        for k, _ in OPERATIONS.items():
            operations.append(OPERATIONS[k]())
        self.operations = operations
        
    def call(self, x, alpha):
        
        a = tf.nn.softmax(alpha)
        out = []
        i = 0
        for op in self.operations:
            out.append(op(x) * a[i])
            i+=1
            
        summed = tf.math.add_n(out)
        
        return summed