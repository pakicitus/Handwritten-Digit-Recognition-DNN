import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Load the trained model
# -----------------------------
model = keras.models.load_model("digit_model.keras")

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Handwritten Digit Recognition")
st.write("Upload a 28×28 handwritten digit and let the neural network predict it!")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

# -----------------------------
# Softmax (Your own implementation)
# -----------------------------
def softmax(logits):

    new_logits = logits - np.max(logits)

    m = logits.shape[1]

    probabilities = np.zeros(m)

    exp_values = np.zeros(m)

    exp_sum = 0

    for i in range(m):

        exp_values[i] = np.exp(new_logits[0, i])

        exp_sum += exp_values[i]

    for i in range(m):

        probabilities[i] = exp_values[i] / exp_sum

    return probabilities

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    # Read image
    img = Image.open(uploaded_file).convert("L")

    # Resize just in case
    img = img.resize((28,28))

    # Display uploaded image
    st.image(img, caption="Uploaded Image", width=180)

    # Convert to numpy
    x_input = np.array(img)

    # Normalize
    x_input = x_input.astype(np.float32)/255.0

    # Uncomment this if your uploaded digits are BLACK on WHITE
    # x_input = 1 - x_input

    # Reshape
    x_input = x_input.reshape(1,784)

    if st.button("Predict"):

        logits = model.predict(x_input)

        probabilities = softmax(logits)

        prediction = np.argmax(logits)

        confidence = probabilities[prediction]*100

        st.success(f"Predicted Digit : {prediction}")

        st.write(f"Model Confidence : **{confidence:.2f}%**")

        st.write("---")

        st.subheader("Confidence for each digit")

        for i in range(10):

            st.write(f"Digit {i} : {probabilities[i]*100:.6f}%")

        # -----------------------------
        # Confidence Graph
        # -----------------------------
        fig, ax = plt.subplots(figsize=(8,4))

        colors = ["steelblue"]*10
        colors[prediction] = "orange"

        ax.bar(range(10),
               probabilities*100,
               color=colors)

        ax.set_xlabel("Digits")

        ax.set_ylabel("Confidence (%)")

        ax.set_title("Prediction Confidence")

        ax.set_xticks(range(10))

        st.pyplot(fig)

        # -----------------------------
        # Raw Logits
        # -----------------------------
        fig2, ax2 = plt.subplots(figsize=(8,4))

        colors = ["steelblue"]*10
        colors[prediction] = "orange"

        ax2.bar(range(10),
                logits.flatten(),
                color=colors)

        ax2.set_xlabel("Digits")

        ax2.set_ylabel("Logits")

        ax2.set_title("Raw Logits Produced by the Neural Network")

        ax2.set_xticks(range(10))

        st.pyplot(fig2)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("📌 Model Details")

    st.write("**Architecture**")

    st.write("784 → 250 → 100 → 10")

    st.write("**Activation**")

    st.write("ReLU")

    st.write("**Output Layer**")

    st.write("Linear (Logits)")

    st.write("**Loss Function**")

    st.write("Sparse Categorical Crossentropy")

    st.write("**Optimizer**")

    st.write("Adam")

    st.write("**Regularization**")

    st.write("L2")

    st.write("**Optimal λ**")

    st.write("0.00001")

    st.write("**Dataset**")

    st.write("MNIST")