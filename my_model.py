import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = keras.models.load_model("digit_model.keras")

st.title("🧠 Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=["png","jpg","jpeg"]
)

from PIL import Image
                                                                                            # Extracting pixels from the image 
img = Image.open("uploaded_file").convert("L")                                              # .convert("L") to convert the image in to two colours black and white

x_input = np.array(img)

x_input = x_input.astype(np.float32) / 255.0
x_input = x_input.reshape(1,784)


def softmax(prediction_final_logits):

    new_logits = prediction_final_logits - np.max(prediction_final_logits)

    m=prediction_final_logits.shape[1]
    prob_array = np.zeros(m)
    logits_sum=0
    logits_exp=np.zeros(m)
    for i in range(m):

        exp_ = np.exp(new_logits[0,i])
        logits_exp[i] = exp_

        logits_sum += exp_

    for i in range(m):
        prob_array[i] = logits_exp[i]/logits_sum

    

    return prob_array 



if st.button("Predict"):
    prediction = model.predict(x_input)
    prediction_final = np.argmax(prediction, axis=1)

    probabilities = softmax(prediction)

    st.success(f"Predicted Digit is : {prediction_final}")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(
        range(10),
        probabilities.flatten()*100
    )

    ax.set_xlabel("Digits")

    ax.set_ylabel("Confidence (%)")

    ax.set_title("Confidence Scores")

    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(
        range(10),
        prediction.flatten()
    )

    ax.set_title("Raw Logits")

    st.pyplot(fig)

with st.sidebar:

    st.header("Model Details")

    st.write("Architecture")

    st.write("784 → 250 →100 →10")

    st.write("Activation : ReLU")

    st.write("Regularization : L2")

    st.write("Optimal λ : 0.00001")

with st.sidebar:

    st.header("Model Details")

    st.write("Architecture")

    st.write("784 → 250 →100 →10")

    st.write("Activation : ReLU")

    st.write("Regularization : L2")

    st.write("Optimal λ : 0.00001")