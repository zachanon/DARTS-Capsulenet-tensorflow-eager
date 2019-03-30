import numpy as np
import tensorflow as tf
import os

from math import pi
from capsules import margin_loss

def train(model, data,
          architecture=None,
          batch_size=64, num_batches=128, epochs=None,
          checkpoint_every=None,
          lr=1e-3, zeta=0.5, cosine=False,
          loss_function=margin_loss):
    
    #unpack data
    (x_train, y_train), (x_val, y_val), _ = data

    num_examples = x_train.shape[0]
    num_val = x_val.shape[0]
    
    global_step = tf.train.get_or_create_global_step()
    checkpoint_directory = "./tmp/training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    
    if architecture is not None:
        checkpoint = tf.train.Checkpoint(model=model, architecture=architecture)
    else:
        checkpoint = tf.train.Checkpoint(model=model)
    
    #precompile the model by calling it
    model(x_train[0:4], architecture)
    
    w_optim = WeightsOptimizer(model.trainable_weights, learning_rate=lr, use_cosine=cosine)
    a_optim = WeightsOptimizer([architecture], learning_rate=zeta, use_cosine=cosine)
    
    loss_library = {
        'train':[],
        'val':[]
    }
    
    grads = []
    
    if epochs is not None:
        num_batches = int(num_examples.value/batch_size)
    else:
        epochs = 1
    
    for epoch in range(epochs):
        for batch in range(num_batches):

            #checkpoint saver
            if checkpoint_every is not None and batch % checkpoint_every == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            #begin weights training pass
            start = batch*batch_size
            end = (batch+1)*batch_size
            x = x_train[start:end]
            y = y_train[start:end]


            with tf.GradientTape() as tape:
                y_hat = model(x, architecture)
                t_loss = loss_function(y, y_hat)

            loss_library['train'].append(t_loss)

            w_grads = tape.gradient(t_loss, model.trainable_weights)
            w_optim.apply_gradients(zip(w_grads, model.trainable_weights))

            
            #end weights training, begin architecture training
            start = int(start*0.2)
            end = int(end*0.2)
            x = x_val[start:end]
            y = y_val[start:end]

            with tf.GradientTape() as tape:
                y_hat = model(x, architecture)
                v_loss = loss_function(y, y_hat)

            loss_library['val'].append(v_loss)

            if architecture is not None:
                a_grads = tape.gradient(v_loss, architecture)
                a_optim.apply_gradients(zip([a_grads],[architecture]))

            print('.', end='')

    return loss_library

class WeightsOptimizer(object):
    """AdamOptimizer with optional cosine cycling"""
    
    def __init__(self, parameters, learning_rate=1e-3, lr_decay=0.95, b1=0.9, b2=0.99, epsilon=1e-8, use_cosine=False):
        
        self.stepsize = learning_rate
        self.decay = lr_decay
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.cosine = use_cosine
        self.timestep = 0
        self.m = []
        self.v = []
        
        for i in range(len(parameters)):
            self.m.append(0)
            self.v.append(0)
      
            
    def apply_gradients(self, grads_and_vars):
        
        self.timestep = self.timestep + 1
        decay = self.decay**self.timestep
        if self.cosine:
            cos_decay = (tf.cos(self.timestep/(pi*self.cosine))+1)/2
            stepsize = self.stepsize*cos_decay*decay
        else:
            stepsize = self.stepsize*decay
        
        for i, (grad, var) in enumerate(grads_and_vars):
            
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*grad
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(grad**2)
            m_hat = self.m[i]/(1-self.b1**self.timestep)
            v_hat = self.v[i]/(1-self.b2**self.timestep)
            update = var - stepsize*m_hat/(tf.sqrt(v_hat)+self.epsilon)
            var.assign(update)