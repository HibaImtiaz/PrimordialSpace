#!/usr/bin/env python
# coding: utf-8


#importing required libraries
import numpy as np
import keras,os
import cv2
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from astroNN.datasets import galaxy10
from tensorflow.keras import utils



def preprocess(images):
    """ converting images from RGB to Grayscale and normalize them
    Parameter
    ---------
    images : list of images
        The images to convert to grayscale

    Return
    -------
        Grayscale images
    """
#     gr_images = []
#     for im in images:
#         img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         gray = cv2.cvtColor(img_gray, cv2.IMREAD_GRAYSCALE)
#         gr_images.append(gray)


#     #converting the list to array
#     img_arr = np.array(gr_images)

    #normalizing the images
    images = images/255

    return images



#loading the dataset
images, labels = galaxy10.load_data()



# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

#converting the arrays to float type
labels = labels.astype(np.float32)
images = images.astype(np.float32)


images = preprocess(images)



images.shape



X = images
y = labels


#splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


#printing the shapes of test and train data
print('Size of X_train', X_train.shape)
print('Size of y_train', y_train.shape)
print('Size of X_test', X_test.shape)
print('Size of y_test', y_test.shape)


#Building the VGG16 Model

model = Sequential()
#first layer
model.add(Conv2D(input_shape=(69,69,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#second layer
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#third layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#fourth layer
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#fifth layer
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#dense layer
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=10, activation="softmax"))

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


model.summary()



checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1,
                      mode='auto')


# #Training

#hist = model.fit(X_train, y_train, steps_per_epoch=50, validation_steps=10,epochs=5,callbacks=[checkpoint,early])
model.fit(X_train, y_train,batch_size=64,epochs=50,validation_data=(X_test,y_test),callbacks=[checkpoint,early])



model.save("astroNN50.h5")
