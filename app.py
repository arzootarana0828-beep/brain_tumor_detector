import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from genai import generate_explanation


MODEL_PATH = "model/brain_tumor_classifier.keras"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="Brain Tumor Detection Using GenAI",
    page_icon="🧠",
    layout="centered"
)


st.title("🧠 Brain Tumor Detection Using GenAI")

st.write(
    "Upload a brain MRI image to obtain an AI classification result "
    "and GenAI educational explanation."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    image_array = np.array(
        image.resize(IMAGE_SIZE)
    ).astype(np.float32)

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        prediction[predicted_index] * 100
    )

    probabilities = {
        CLASS_NAMES[i]: float(prediction[i] * 100)
        for i in range(len(CLASS_NAMES))
    }


    st.subheader("🔍 AI Prediction")

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )


    st.subheader("📊 Class Probabilities")

    for name, probability in probabilities.items():

        st.write(
            f"**{name}: {probability:.2f}%**"
        )

        st.progress(
            min(float(probability / 100), 1.0)
        )


    st.subheader("🤖 GenAI Explanation")


    if st.button("Generate Educational Explanation"):

        with st.spinner("Generating explanation..."):

            explanation = generate_explanation(
                predicted_class,
                confidence,
                probabilities
            )

        st.write(explanation)