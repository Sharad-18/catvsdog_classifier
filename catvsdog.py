import streamlit as st
from PIL import Image  # Import the Image module from PIL library
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("dog_cat_classifier.h5")
 
st.title("Dog vs. Cat Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = tf.image.decode_image(uploaded_file.getvalue(), channels=3)
    image = tf.image.resize(image, [256, 256])  # Resize the image to match model input size
    image = tf.cast(image, tf.float32) / 255.  # Normalize the image
    
    # Make predictions
    prediction = model.predict(tf.expand_dims(image, axis=0))
    if prediction[0][0] > 0.4:
        st.write("Prediction: Dog")
    else:
        st.write("Prediction: Cat")
