import os, gdown

import tensorflow as tf

def load_file():
    gdown.download('https://drive.google.com/uc?id=1YSVfAQT8rN8gGo3tvbXGzBnGXmp0Ojs_', "model.zip")
    os.remove("/app/helper/model.zip")

def load_model():
    if not os.path.isdir("/app/helper/modelapi"):
        load_file()
    model = tf.keras.models.load_model()
    return model

def pinpoint(image):
    model = load_model()
    result = model.predict(image)
    return result