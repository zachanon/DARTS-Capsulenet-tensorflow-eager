import keras
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist



def load_mnist(flat=True, val_split=0.2):
    """
    returns tuple (x_train, y_train), (x_test, y_test) """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    num_val = int(x_train.shape[0] * val_split)
    
    x_train = x_train.reshape(x_train.shape[0], 784) / 255
    x_test = x_test.reshape(x_test.shape[0], 784) / 255
    
    if flat is not True:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    
    x_val = x_train[:num_val]
    x_train = x_train[num_val:]
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_val = y_train[:num_val]
    y_train = y_train[num_val:]
    
   
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_fashion_mnist():
    
    """
    returns tuple (x_train, y_train), (x_test, y_test) """
    
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)