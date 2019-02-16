from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import cv2, numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

def CNN_16(weights_path=None, shape=(48, 48)):
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape=(1,48,48)))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    # pool_size is generally (2,2). Will reduce size of feature maps and divide it by 2.
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # Take all pooled feature maps and put in into one huge vector(Spatial information preserved)
    model.add(Flatten())
    
    # Dense is used to add fully connected layer.
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    print ("Create model successfully")
    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer='adam', loss='categorical_crossentropy', \
        metrics=['accuracy'])

    return model
