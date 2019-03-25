import numpy as np
import tensorflow as tf

from capsules import margin_loss

def train(model, data, architecture, batch_size=64, num_batches=128, epochs=None, lr=1e-3, zeta=0.5):
    
    #unpack data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data

    input_shape = x_train[0].shape
    num_class = 10
    num_examples = x_train.shape[0]
    num_val = x_val.shape[0]
    
    global_step = tf.train.get_or_create_global_step()
    
    w_optim = tf.train.AdamOptimizer(learning_rate=lr)
    a_optim = tf.train.GradientDescentOptimizer(learning_rate=zeta)
    
    loss_library = {
        'train':[],
        'val':[]
    }
    
    grads = []
    
    if epochs is not None:
        num_batches = int(num_examples.value/batch_size) * epochs
    
    for batch in range(num_batches):
        
        #begin weights training pass
        start = batch*batch_size
        end = (batch+1)*batch_size
        x = x_train[start:end]
        y = y_train[start:end]
        
        
        ###TODO: fix adam for eager
        with tf.GradientTape() as tape:
            y_hat = model(x, architecture)
            t_loss = margin_loss(y, y_hat)
            
        loss_library['train'].append(t_loss)
        
        w_grads = tape.gradient(t_loss, model.trainable_weights)
        w_optim.apply_gradients(zip(w_grads, model.trainable_weights), global_step=global_step)
                
        #end weights training, begin architecture training
        
        start = int(start*0.2)
        end = int(end*0.2)
        x = x_val[start:end]
        y = y_val[start:end]
        
        with tf.GradientTape() as tape:
            y_hat = model(x, architecture)
            v_loss = margin_loss(y, y_hat)
            
        loss_library['val'].append(v_loss)
            
        a_grads = tape.gradient(v_loss, architecture)
        new_arch = architecture - zeta*a_grads
        architecture.assign(new_arch)
        
        
        print('.', end='')
        
    return loss_library