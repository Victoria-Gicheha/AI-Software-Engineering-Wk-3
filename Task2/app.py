import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load your trained MNIST model

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()


# App Title

st.title("🧠 MNIST Handwritten Digit Classifier")
st.write("Upload an image of a digit (0–9) and let the model predict it!")


# Upload Section

uploaded_file = st.file_uploader("📤 Upload a digit image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # invert colors if background is white
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Display results
    st.subheader(f"✅ Predicted Digit: {predicted_class}")
    st.bar_chart(prediction[0])
else:
    st.info("👆 Please upload an image to start the prediction.")
