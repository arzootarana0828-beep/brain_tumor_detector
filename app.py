from pathlib import Path
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from google import genai

MODEL_PATH = Path("model/brain_tumor_classifier.keras")
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to get the model prediction.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    return array


@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def generate_explanation(predicted_class, score):
    client = get_gemini_client()

    if client is None:
        return "Gemini explanation is unavailable because the API key is not set."

    prompt = f"""
You are helping explain the output of an educational brain MRI
machine-learning project.

The machine-learning model predicted:
Class: {predicted_class}
Model score: {score:.1%}

Explain this result in simple language for a student.

Important:
- Do not diagnose the person.
- Do not say that a tumor is definitely present or absent.
- Explain that this is only a machine-learning model prediction.
- Mention that medical images must be reviewed by a qualified radiologist
  or clinician for medical interpretation.
- Keep the explanation short and easy to understand.
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return interaction.output_text

    except Exception as error:
        return f"Gemini explanation could not be generated: {error}"


if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.info("Run: python train.py --data-dir data")
    st.stop()


uploaded = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)


if uploaded is not None:
    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded MRI image",
        use_container_width=True
    )

    model = load_model()

    input_image = prepare_image(image)

    scores = model.predict(
        input_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(scores))
    predicted_class = CLASS_NAMES[predicted_index]
    predicted_score = float(scores[predicted_index])

    st.subheader("Model Output")

    st.metric(
        "Predicted Class",
        predicted_class.title()
    )

    st.metric(
        "Model Probability",
        f"{predicted_score:.1%}"
    )

    st.subheader("Class Probabilities")

    for class_name, score in zip(CLASS_NAMES, scores):
        st.write(
            f"**{class_name.title()}** — {float(score):.1%}"
        )

        st.progress(float(score))

    st.subheader("AI Explanation")

    if st.button("Generate Explanation with Gemini"):
        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(
                predicted_class,
                predicted_score
            )

        st.write(explanation)