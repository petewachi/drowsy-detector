import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from keras_visualizer import visualizer 


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(48,48)
train_batch= generator('ProjectCode\Dataset\drowsiness_dataset\MouthYawn\Train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('ProjectCode\Dataset\drowsiness_dataset\MouthYawn\Test',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)


# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])

opt = optimizers.Adam(learning_rate=0.0002,epsilon=5e-8)
#opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

N = 30
H = model.fit(train_batch, validation_data=valid_batch,epochs=N,steps_per_epoch=SPE ,validation_steps=VS)

model.save('ProjectCode\models\MouthYawnClassifier.h5', overwrite=True)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Mouth Yawn: Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
#plt.savefig("plot.png")

try:
    utils.plot_model(model, to_file='ProjectCode\models\MouthYawnClassifier_model_arch.png', show_shapes=True, show_layer_names=True)
except:
    pass

"""
plt.plot(model.history['loss'], label='loss')
plt.plot(model.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(model.history['accuracy'], label='acc')
plt.plot(model.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
"""