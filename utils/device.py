import tensorflow as tf

def get_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return "gpu"
    else:
        return "cpu"