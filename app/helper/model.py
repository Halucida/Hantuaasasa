import os, gdown

import tensorflow as tf

def load_file():
    gdown.download('https://drive.google.com/uc?id=1bC2OJyC-j8Lr4Jtk73DGZo0qeWPQPUuu', "model.zip")
    os.remove("/app/helper/model.zip")

def load_model():
    if not os.path.isdir("/app/helper/modelapi"):
        load_file()
    model = tf.keras.models.load_model()
    return model

@tf.function
def alignment_helper(image):
    img = tf.io.read_file(image)
    return img

def pinpoint(rimage):
    model = load_model()
    image = alignment_helper(rimage)
    hasil = model.predict(image)
    return hasil