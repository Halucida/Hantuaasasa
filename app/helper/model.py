import os, gdown

import tensorflow as tf

from helper.image import alignment_helper

def load_file():
    gdown.download('https://drive.google.com/uc?id=1YSVfAQT8rN8gGo3tvbXGzBnGXmp0Ojs_', "model.zip")
    os.remove("/app/helper/model.zip")

def load_model():
    if not os.path.isdir("/app/helper/modelapi"):
        load_file()
    model = tf.keras.models.load_model()
    return model

def pinpoint(rimage):
    model = load_model()
    image = alignment_helper(rimage)
    hasil = model.predict(image)
    return hasil