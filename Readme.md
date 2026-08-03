# 🧠 Handwritten Digit Recognition using Deep Neural Networks
![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)
> An end-to-end Deep Learning project exploring neural network design, overfitting, regularization, hyperparameter tuning, and handwritten digit recognition using the MNIST dataset.

---

## 📖 About the Project

This project was built to understand **how Deep Neural Networks actually learn**, rather than simply training a model and reporting its accuracy.

Instead of stopping after creating a classifier, I explored the complete model development process by experimenting with different neural network architectures, analyzing overfitting, applying L2 regularization, tuning hyperparameters, and evaluating how each decision affected the model's ability to generalize.

The notebook follows the same workflow used in many real machine learning projects—from data preprocessing to model evaluation and finally deploying the trained model for interactive handwritten digit prediction.

---

## 🚀 Features

- 📂 Load the MNIST dataset directly from the original IDX files
- 🧠 Build multiple Deep Neural Network architectures using TensorFlow/Keras
- 📉 Analyze overfitting using Training and Cross Validation loss
- ⚖️ Compare a Complex Model with a Simpler Model
- 🛡️ Reduce overfitting using L2 Regularization
- 🔍 Perform Lambda (λ) Hyperparameter Search
- 📊 Compare models using Classification Error instead of relying only on Loss
- 📈 Visualize Training & Validation Loss Curves
- ❌ Display Misclassified Test Images for Error Analysis
- 🎨 Interactive Handwritten Digit Predictor
- 📊 Display Confidence Scores and Raw Logits for every prediction

---

# 🏗 Project Workflow

```
Raw MNIST Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Complex Neural Network
        │
        ▼
Overfitting Analysis
        │
        ▼
Simpler Neural Network
        │
        ▼
L2 Regularization
        │
        ▼
Lambda Hyperparameter Search
        │
        ▼
Final Model Selection
        │
        ▼
Test Set Evaluation
        │
        ▼
Interactive Digit Prediction
```

---

## 🧠 Neural Network Architectures

### Complex Model

```
784
 ↓
Dense(250, ReLU)
 ↓
Dense(100, ReLU)
 ↓
Dense(10)
```

---

### Simpler Model

```
784
 ↓
Dense(100, ReLU)
 ↓
Dense(35, ReLU)
 ↓
Dense(10)
```

---

### Final Regularized Model

```
784
 ↓
Dense(250, ReLU, L2)
 ↓
Dense(100, ReLU, L2)
 ↓
Dense(10)
```

The final model was selected after comparing multiple values of the regularization parameter (λ) using Cross Validation Classification Error.

---

# 📊 Project Highlights

Throughout this project I explored several important Deep Learning concepts, including:

- Deep Neural Networks
- ReLU Activation
- Softmax
- Sparse Categorical Cross Entropy
- Adam Optimizer
- L2 Regularization
- Hyperparameter Tuning
- Train / Cross Validation / Test Split
- Model Generalization
- Error Analysis
- Confidence Scores
- Logits

---

# ✍ Interactive Prediction

The notebook also includes an interactive handwritten digit recognizer.

Simply:

1. Open **Draw_here.png**
2. Draw a digit using Microsoft Paint
3. Save the image
4. Run the prediction cell

The notebook displays:

- Predicted digit
- Confidence for every class
- Raw logits
- Logit visualization

---

# 📸 Notebook Includes

- 📈 Training & Validation Loss Curves
- 📊 Lambda Search Results
- 📉 Model Comparison
- ❌ Wrong Prediction Visualization
- 🎨 Interactive Handwritten Digit Prediction
- 📊 Logits Bar Graph

---

# 🛠 Technologies Used

- Python
- NumPy
- TensorFlow
- Keras
- Matplotlib
- Pillow
- Scikit-learn

---

# 🎯 Key Learning Outcomes

This project helped me move beyond simply using neural networks and focus on understanding **why models behave the way they do**.

Some of my biggest takeaways include:

- Detecting and reducing overfitting
- Choosing appropriate model complexity
- Understanding the effect of L2 Regularization
- Selecting hyperparameters using Cross Validation
- Evaluating models using Classification Error
- Understanding logits and Softmax
- Performing error analysis using misclassified images

---

# 🚀 Future Improvements

- Deploy as a web application using Streamlit
- Draw digits directly inside the browser
- Add a Confusion Matrix
- Experiment with CNN architectures
- Compare DNN vs CNN performance

---

## 👨‍💻 Author

**PAKICITUS**

This project was built as part of my Deep Learning learning journey. The primary goal was not only to build an accurate handwritten digit classifier but also to understand the reasoning behind each design choice through experimentation and analysis.