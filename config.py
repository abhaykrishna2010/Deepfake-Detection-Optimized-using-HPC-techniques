import tensorflow as tf

class Config:
    CASIA1 = "C:/Users/abhay/OneDrive/Documents/VIT/Sem 7/HPC/Project/CASIA1"
    CASIA2 = "C:/Users/abhay/OneDrive/Documents/VIT/Sem 7/HPC/Project/CASIA2"
    autotune = tf.data.experimental.AUTOTUNE
    epochs = 30
    batch_size = 32
    lr = 1e-3
    name = 'xception'
    n_labels = 2
    image_size = (224, 224)
    decay = 1e-6
    momentum = 0.95
    nesterov = False
