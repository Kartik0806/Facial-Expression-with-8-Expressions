import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py

st.header('Emotion Detector')

def main():
    file_uploaded=st.file_uploader('Choose the file', type=['jpg','jpeg','jpg'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result=predict_class(image)
        st.write(result)
        st.pyplot(figure)
def predict_class(image):
    model=load_model('FER.h5')
    shape=((256,256,3))
    test_image=image.resize((256,256))
    test_image=preprocessing.image.img_to_array(test_image)
    test_image=test_image/255.0
    test_image=np.expand_dims(test_image,axis=0)
    class_names=['anger',
                 'surprise',
                 'disgust',
                 'fear',
                 'neutral',
                 'happiness',
                 'sadness',
                 'contempt']
    prediction=model.predict(test_image)
    scores=tf.nn.softmax(prediction[0])
    scores=scores.numpy()
    image_class=class_names[np.argmax(scores)]
    result='The emotion shown by this person is: {}'.format(image_class)
    return result

if __name__=="__main__":
    main()