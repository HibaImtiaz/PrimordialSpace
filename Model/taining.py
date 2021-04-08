import numpy as np
import keras,os
import cv2
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from astroNN.datasets import galaxy10
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def preprocess(images):
    """ normalize the images
    Parameter
    ---------
    images : list of images
        The images to convert to normalize

    Return
    -------
        normalized Images
    """

    #normalizing the images
    images = images/255



#loading the dataset
images, labels = galaxy10.load_data()

# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

preprocess(images)

X = images
y = labels


#splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#printing the shapes of test and train data
print('Size of X_train', X_train.shape)
print('Size of y_train', y_train.shape)
print('Size of X_test', X_test.shape)
print('Size of y_test', y_test.shape)



#converting the arrays to float type
labels = labels.astype(np.float32)
images = images.astype(np.float32)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
input_shape=(69,69,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


model.compile(
loss='categorical_crossentropy',
optimizer='Adam',
metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5,
validation_data=(X_test, y_test))


output = model.predict(X_train[0])

print("Model prediction:",output)


# model.save("astroNN20.h5")

plt.plot(model.history.history['loss'],color='b',
label='Training Loss')
plt.plot(model.history.history['val_loss'],color='r',
label='Validation Loss')
plt.legend()
plt.show()


plt.plot(model.history.history['accuracy'],color='b',
label='Training  Accuracy')
plt.plot(model.history.history['val_accuracy'],color='r',
label='Validation Accuracy')
plt.legend()
plt.show()
