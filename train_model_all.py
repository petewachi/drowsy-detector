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

isTrainClosedEyes = True
isTrainMouthYawn = True
isTrainFaceYawn = True

############################################################################################################################################################################################
#Closed Eyes Model
############################################################################################################################################################################################
if isTrainClosedEyes:
    BS= 32
    TS=(24,24)
    path = 'ProjectCode\Dataset\drowsiness_dataset\ClosedEye'
    #path = 'ProjectCode\Dataset\drowsiness_dataset\MRLClosedEyes'
    train_batch= generator(path+'\Train',shuffle=True, batch_size=BS,target_size=TS)
    valid_batch= generator(path+'\Test',shuffle=True, batch_size=BS,target_size=TS)
    SPE= len(train_batch.classes)//BS
    VS = len(valid_batch.classes)//BS
    print(SPE,VS)


    # img,labels= next(train_batch)
    # print(img.shape)

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
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
        Dropout(0.2),
    #flatten since too many dimensions, we only want a classification output
        Flatten(),
    #fully connected to get all relevant data
        Dense(128, activation='relu'),
    #one more dropout for convergence' sake :) 
        Dropout(0.5),
    #output a softmax to squash the matrix into output probabilities
        Dense(2, activation='softmax')
    ])

    opt = optimizers.Adam(learning_rate=0.0001, epsilon=5e-8)
    #opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

    N = 30
    closedEyeClassifier = model.fit(train_batch, validation_data=valid_batch,epochs=N,steps_per_epoch=SPE ,validation_steps=VS)

    model.save('ProjectCode\models\Temp\closedEyeClassifier.h5', overwrite=True)

    try:
        utils.plot_model(model, to_file='ProjectCode\models\Temp\closedEyeClassifier_model_arch.png', show_shapes=True, show_layer_names=True)
    except:
        pass

############################################################################################################################################################################################
#Mouth Yawn Model
############################################################################################################################################################################################
if isTrainMouthYawn:
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
    MouthYawnClassifier = model.fit(train_batch, validation_data=valid_batch,epochs=N,steps_per_epoch=SPE ,validation_steps=VS)

    model.save('ProjectCode\models\Temp\MouthYawnClassifier.h5', overwrite=True)


    try:
        utils.plot_model(model, to_file='ProjectCode\models\Temp\closedEyeClassifier_model_arch.png', show_shapes=True, show_layer_names=True)
    except:
        pass


############################################################################################################################################################################################
#Face Yawn Model
############################################################################################################################################################################################
if isTrainFaceYawn:
    BS= 32
    TS=(48,48)
    train_batch= generator('ProjectCode\Dataset\drowsiness_dataset\FaceYawn\Train',shuffle=True, batch_size=BS,target_size=TS)
    valid_batch= generator('ProjectCode\Dataset\drowsiness_dataset\FaceYawn\Test',shuffle=True, batch_size=BS,target_size=TS)
    SPE= len(train_batch.classes)//BS
    VS = len(valid_batch.classes)//BS
    print(SPE,VS)



    # img,labels= next(train_batch)
    # print(img.shape)

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(pool_size=(2, 2)),
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
        Dropout(0.6),
    #output a softmax to squash the matrix into output probabilities
        Dense(2, activation='softmax')
    ])

    opt = optimizers.Adam(learning_rate=0.0002, epsilon=5e-8)
    #opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

    N = 30
    FaceYawnClassifier = model.fit(train_batch, validation_data=valid_batch,epochs=N, steps_per_epoch=SPE ,validation_steps=VS)

    model.save('ProjectCode\models\Temp\FaceYawnClassifier.h5', overwrite=True)

    try:
        utils.plot_model(model, to_file='ProjectCode\models\Temp\closedEyeClassifier_model_arch.png', show_shapes=True, show_layer_names=True)
    except:
        pass

############################################################################################################################################################################################
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure(1)
plt.subplot(211)
if isTrainClosedEyes:
    plt.plot(np.arange(0, N), closedEyeClassifier.history["loss"], label="closedEye_train_loss")
    plt.plot(np.arange(0, N), closedEyeClassifier.history["val_loss"], label="closedEye_val_loss")
if isTrainMouthYawn:
    plt.plot(np.arange(0, N), MouthYawnClassifier.history["loss"], label="mouthYawn_train_loss")
    plt.plot(np.arange(0, N), MouthYawnClassifier.history["val_loss"], label="mouthYawn_val_loss")
if isTrainFaceYawn:
    plt.plot(np.arange(0, N), FaceYawnClassifier.history["loss"], label="faceYawn_train_loss")
    plt.plot(np.arange(0, N), FaceYawnClassifier.history["val_loss"], label="faceYawn_val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
#plt.show()
#plt.savefig("plot.png")

plt.style.use("ggplot")
plt.figure(1)
plt.subplot(212)
if isTrainClosedEyes:
    plt.plot(np.arange(0, N), closedEyeClassifier.history["accuracy"], label="closedEye_train_accuracy")
    plt.plot(np.arange(0, N), closedEyeClassifier.history["val_accuracy"], label="closedEye_val_accuracy")
if isTrainMouthYawn:
    plt.plot(np.arange(0, N), MouthYawnClassifier.history["accuracy"], label="mouthYawn_train_accuracy")
    plt.plot(np.arange(0, N), MouthYawnClassifier.history["val_accuracy"], label="mouthYawn_val_accuracy")
if isTrainFaceYawn:
    plt.plot(np.arange(0, N), FaceYawnClassifier.history["accuracy"], label="faceYawn_train_accuracy")
    plt.plot(np.arange(0, N), FaceYawnClassifier.history["val_accuracy"], label="faceYawn_val_accuracy")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("Traing Loss and Accuracy.png")
plt.show()



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