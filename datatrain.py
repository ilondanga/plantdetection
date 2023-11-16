import keras
import pathlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import  layers
from tensorflow.keras.optimizers.legacy import Adam


path ="/home/filtronic/plantdetect/data"
dataset =pathlib.Path(path)
batch_size = 32
height = 180
width = 180

training_ds =tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split= 0.2,
    subset="training",
    seed=123,
    image_size= (height,width),

)
validation_ds =tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split= 0.2,
    subset="validation",
    seed=123,
    image_size=(height,width),
    batch_size=batch_size
)
class_names = training_ds.class_names
ATUTUNE = tf.data.AUTOTUNE

train_ds = training_ds.cache().prefetch(buffer_size=ATUTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=ATUTUNE)
classes = 38

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1/255, input_shape=(height,width,3)),
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation ="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation ="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),

    layers.Dense(512,activation ="softmax"),
    layers.Dense(classes)


])

model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
model.fit(train_ds,epochs=4,validation_data=val_ds)
model.save("plants_model.h5")