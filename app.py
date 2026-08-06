import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# ==========================================
# Load Model
# ==========================================

model = keras.models.load_model("digit_model.keras")

# ==========================================
# Softmax (Your own implementation)
# ==========================================

def softmax(logits):

    new_logits = logits - np.max(logits)

    m = logits.shape[1]

    prob_array = np.zeros(m)

    logits_sum = 0

    logits_exp = np.zeros(m)

    for i in range(m):

        exp_ = np.exp(new_logits[0, i])

        logits_exp[i] = exp_

        logits_sum += exp_

    for i in range(m):

        prob_array[i] = logits_exp[i] / logits_sum

    return prob_array


# ==========================================
# Title
# ==========================================

st.title("🧠 Handwritten Digit Recognition using Neural Networks")

st.write(
    "Draw a digit (0-9) in the canvas below and click **Predict**."
)

# ==========================================
# Canvas
# ==========================================

canvas_result = st_canvas(

    fill_color="black",

    stroke_width=35,

    stroke_color="white",

    background_color="black",

    width=280,

    height=280,

    drawing_mode="freedraw",

    key="canvas"

)

# ==========================================
# Prediction
# ==========================================

if st.button("Predict"):

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        img = Image.fromarray(canvas_result.image_data.astype("uint8"))

        img = img.convert("L")

        img = img.resize((28,28))

        st.subheader("Processed 28×28 Image")

        st.image(img,width=120)

        x_input = np.array(img)

        x_input = x_input.astype(np.float32)/255.0

        x_input = x_input.reshape(1,784)

        prediction_logits = model.predict(x_input)

        probabilities = softmax(prediction_logits)

        prediction = np.argmax(probabilities)

        confidence = probabilities[prediction]*100

        st.success(f"Predicted Digit : {prediction}")

        st.write(f"### Confidence : {confidence:.2f}%")

        st.write("---")

        st.subheader("Confidence for Every Digit")

        for i in range(10):

            st.write(f"Digit {i} : {probabilities[i]*100:.6f}%")

        # ======================================
        # Confidence Graph
        # ======================================

        # fig, ax = plt.subplots(figsize=(8,4))

        # colors = ["steelblue"]*10

        # colors[prediction]="orange"

        # ax.bar(

        #     range(10),

        #     probabilities*100,

        #     color=colors

        # )

        # ax.set_xticks(range(10))

        # ax.set_xlabel("Digits")

        # ax.set_ylabel("Confidence (%)")

        # ax.set_title("Prediction Confidence")

        # st.pyplot(fig)

        # # ======================================
        # # Raw Logits Graph
        # # ======================================

        # fig2, ax2 = plt.subplots(figsize=(8,4))

        # colors = ["steelblue"]*10

        # colors[prediction]="orange"

        # ax2.bar(

        #     range(10),

        #     prediction_logits.flatten(),

        #     color=colors

        # )

        # ax2.set_xticks(range(10))

        # ax2.set_xlabel("Digits")

        # ax2.set_ylabel("Logits")

        # ax2.set_title("Raw Logits Produced by the Neural Network")

        # st.pyplot(fig2)


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.header("Model Details")

    st.write("### Architecture")

    st.write("784 → 250 → 100 → 10")

    st.write("### Hidden Activation")

    st.write("ReLU")

    st.write("### Output")

    st.write("Linear (Logits)")

    st.write("### Regularization")

    st.write("L2")

    st.write("Optimal λ = 0.00001")

    st.write("### Optimizer")

    st.write("Adam")

    st.write("### Dataset")

    st.write("MNIST")

    st.write("### Author")

    st.write("Harsh Vats")