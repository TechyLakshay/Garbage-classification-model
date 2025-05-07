import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('garbage_classification_model.h5')

# Class names
class_names = ['paper', 'cardboard', 'biological', 'metal', 'plastic', 
               'green-glass', 'brown-glass', 'white-glass', 'clothes', 
               'shoes', 'batteries', 'trash']

# Streamlit app
st.title("🗑️ Garbage Classification App")
st.write("Upload an image of garbage to predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🛠️ Step 1: Load the uploaded image
    img = Image.open(uploaded_file)
    
    # 🛠️ Step 2: Convert to RGB to ensure 3 channels
    img = img.convert('RGB')
    
    # 🛠️ Step 3: Display image in Streamlit
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # 🛠️ Step 4: Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 🛠️ Step 5: Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # 🛠️ Step 6: Show result
    st.success(f"Predicted Class: **{predicted_class}**")
