from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from astroNN.datasets.galaxy10 import galaxy10cls_lookup
import numpy as np

model = load_model("astroNN.h5")


def preprocess(img):
  img = img_to_array(img)
  img = img/255
  img = np.expand_dims(img,0)
  return img

def predict(location):
  img = load_img(location,target_size=(69,69))
  img = preprocess(img)
  output = model.predict(img)
  predictedClass = np.argmax(output[0])
  pred_made = galaxy10cls_lookup(predictedClass)
  return pred_made
