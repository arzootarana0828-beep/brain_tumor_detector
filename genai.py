import os
import time
from google import genai


def generate_explanation(prediction, confidence, probabilities):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ Gemini API key is not configured."

    client = genai.Client(api_key=api_key)

    probability_text = "\n".join(
        f"{name}: {probability:.2f}%"
        for name, probability in probabilities.items()
    )

    prompt = f"""
You are an educational AI assistant for a brain MRI image-classification project.

The TensorFlow model produced:

Predicted class: {prediction}
Model confidence: {confidence:.2f}%

Class probabilities:
{probability_text}

Explain this result in simple educational language.

Include:
1. What the predicted class generally means.
2. What the confidence score means.
3. Why an AI prediction should not be treated as a medical diagnosis.
4. A short recommendation to consult a qualified medical professional.

Do not diagnose the person.
Keep the explanation concise.
"""

    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash"
    ]

    for model_name in models:
        for attempt in range(2):
            try:
                interaction = client.interactions.create(
                    model=model_name,
                    input=prompt
                )

                return interaction.output_text

            except Exception as e:
                error_text = str(e)

                if "503" in error_text or "UNAVAILABLE" in error_text:
                    if attempt == 0:
                        time.sleep(5)
                        continue

                break

    return (
        "⚠️ Gemini is temporarily unavailable.\n\n"
        "The TensorFlow prediction is still available above. "
        "Please try generating the explanation again later."
    )