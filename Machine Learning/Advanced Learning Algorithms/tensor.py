import tensorflow as tf
import numpy as np

w = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = w*w
    x = tape.gradient(y,w)



print(x)






