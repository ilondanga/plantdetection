import tensorflow as tf
import numpy as np
import keras.preprocessing as kr
import keras
import keras_preprocessing

height = 180
width = 180

class_names=['Gray_leaf_Spot','Healthy','Blight','Common_Rust',]


model=keras.models.load_model('plants_model.h5')

path="/home/filtronic/plantdetect/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/Corn_(maize)___Common_rust_/RS_Rust 1564_flipLR.JPG"
img=kr.image.load_img(path,target_size=(height,width))
img=kr.image.img_to_array(img)
img=tf.expand_dims(img,0)
print(img.shape)

predictions=model.predict(img)

score=tf.nn.softmax(predictions[0])
print("THis plant is {} with a{:2f} percent confidence."
      .format(class_names[np.argmax(score)],100*np.max(score)))
