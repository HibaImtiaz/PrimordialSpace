import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from astroNN.datasets import galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup

images, labels = galaxy10.load_data()

model = load_model("astroNN.h5")

def preprocess(img):
  img = img/255
  img = np.expand_dims(img,0)
  return img

img = load_img("gal.jpg",target_size=(69,69))
img_array = img_to_array(img)

print(type(img))
print(img.format)
print(img.mode)
print(img.size)

img = preprocess(img_array)

img1 = images[10]

def pred(img):
  # img = np.expand_dims(img,0) # <--- add batch axis
  output = model.predict(img)
  predictedClass = np.argmax(output[0])
  pred_made = galaxy10cls_lookup(predictedClass)
  print("Prediction: ", pred_made)

pred(img)
