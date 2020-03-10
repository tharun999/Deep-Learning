


import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,MaxPooling2D,Convolution2D
model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
#second convolution layer
model.add(Convolution2D(32,(3,3),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(activation="relu",units=128))
model.add(Dense(activation="sigmoid",units=1))#if we have categorical output then we go with softmax activation function and for binary we use sigmoid
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#if we have more categories to classify then we go with categorical_crossentropy loss Function

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('chest/pneumonia/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('chest/pneumonia/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
model.fit_generator(training_set,samples_per_epoch=3513,nb_epoch=3,validation_data=test_set,nb_val_samples=1171)
#single prediction
import numpy as np
from keras.preprocessing import image
test=image.load_img("chest/pneumonia/single/no.jpeg",target_size=(64,64))
test=image.img_to_array(test)
test=np.expand_dims(test,axis=0)
res=model.predict(test)
training_set.class_indices
if res[0][0]==1:
    print("Pneumonia")
else:
    print("Normal")
        
