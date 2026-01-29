import streamlit as st
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image

model = ResNet50(weights='imagenet')

st.title("Predict pictures")

uploaded_pic = st.file_uploader(
    "Upload a picture",type=["jpg", "jpeg", "png"]
)

if uploaded_pic is not None:
    img = Image.open(uploaded_pic)

    st.image(img, caption="Uploaded picture", width="stretch")

    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)


    preds = model.predict(x)

    st.subheader("Prediction:")
    for label, desc, prob in decode_predictions(preds, top=3)[0]:
        st.write(f"**{desc}** – {prob:.2f}")

    
