import os
import base64  # for csv exports
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


def load_model():
    """Loads the pre-trained model."""
    model_path = 'outputs/v1/mildew_detection_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model


def predict_mildew(model, img_array):
    """Predicts if a cherry leaf has powdery mildew and returns the
    confidence level."""
    predictions = model.predict(img_array)
    confidence = predictions[0][0]

    if confidence < 0.5:
        confidence = 1 - confidence
        prediction_class = "Healthy"
    else:
        prediction_class = "Powdery Mildew"

    confidence_percentage = "{:.2f}%".format(confidence * 100)
    return prediction_class, confidence_percentage


def resize_image(image, target_size=(224, 224)):
    """Resizes an image to the target size."""
    return image.resize(target_size)


def mildew_powdery_detection():
    st.title("Powdery Mildew on Cherry Leaves Detection")
    st.info(
        "The client is interested in predicting if a cherry leaf is "
        "healthy or contains powdery mildew."
    )

    st.write(
        "* You can download a set of healthy and powdery mildew images for "
        "live prediction. You can download the images from "
        "[here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    model = load_model()
    uploaded_files = st.file_uploader(
        "Choose cherry leaf images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    results_list = []

    for uploaded_file in uploaded_files:
        with st.container():
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_container_width=True)

            resized_img = resize_image(img)
            img_array = np.expand_dims(
                np.array(resized_img) / 255.0,
                axis=0
            )

            prediction, confidence = predict_mildew(model, img_array)
            st.write(f"Prediction: **{prediction}**")
            st.write(f"Confidence: **{confidence}**")

            results_list.append({
                "Image Name": uploaded_file.name,
                "Prediction": prediction,
                "Confidence": confidence
            })

    if results_list:
        st.write("## Summary of Prediction Results")
        results_df = pd.DataFrame(results_list)
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download prediction results as CSV",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv'
        )


if __name__ == "__main__":
    mildew_powdery_detection()