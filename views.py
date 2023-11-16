
from flask import Flask, request, jsonify, send_from_directory
import os

import tensorflow as tf
import numpy as np
import keras.preprocessing as kr
import keras
import keras_preprocessing

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'


height = 180
width = 180

class_names=['Blight','Common_Rust','Gray_leaf_Spot','Healthy']
model=keras.models.load_model('plants_model.h5')

path="75080070-have-disease-of-maize-leaves.jpg"

img=kr.image.load_img(path,target_size=(height,width))
img=kr.image.img_to_array(img)
img=tf.expand_dims(img,0)
print(img.shape)

predictions=model.predict(img)

score=tf.nn.softmax(predictions[0])
print("THis plant is {} with a{:2f} percent confidence."
      .format(class_names[np.argmax(score)],100*np.max(score)))

class plant_upload():
    def upload(self):
        pass

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']

    # Save the image file to the uploads folder
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    # Make a prediction on the image
    predictions = model.predict(image_path)

    # Return the predictions as JSON
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
