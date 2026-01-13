import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model

# ---------------------------
# Title
# ---------------------------
st.title("🐱🐶 Cats vs Dogs Classifier*****************")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_cnn():
    model = load_model("cats_dogs_model.keras", compile=False)
    model(tf.zeros((1, 150, 150, 3)))  # Build model
    return model

model = load_cnn()

# ---------------------------
# Upload image
# ---------------------------
file = st.file_uploader(
    "Upload an image of a cat or dog",
    type=["jpg", "jpeg", "png"]
)

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # Preprocess
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Result
    st.subheader("Prediction:")
    if prediction > 0.5:
        st.success(f"🐶 Dog ({prediction:.2f})")
    else:
        st.success(f"🐱 Cat ({1 - prediction:.2f})")
