from keras.models import load_model
from tensorflow.keras import utils
from keras_visualizer import visualizer 
import visualkeras
import matplotlib.pyplot as plt 
import numpy as np
path = 'ProjectCode\models'
eye_model = load_model(path+'\ClosedEyeClassifier.h5')
face_yawn_model = load_model(path+'\FaceYawnClassifier.h5')
mouth_yawn_model = load_model(path+'\MouthYawnClassifier.h5')

model = eye_model
dot_img_file = '/tmp/model_1.png'
utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True, show_layer_activations = True)

#visualkeras.layered_view(model, legend=True) # without custom font
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 12)

#ClosedEyeModel
#MouthYawnModel
#FaceYawnModel
visualkeras.layered_view(model, to_file=path+'\ClosedEyeModel.png', legend=True, font=font) # selected font



N = 30
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), model.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), model.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), model.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), model.history["val_accuracy"], label="val_acc")
plt.title("Closed Eyes: Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
#visualizer(model, format='png', view=True)
