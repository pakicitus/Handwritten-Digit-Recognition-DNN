#%%
import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = keras.models.load_model("digit_model.keras")
import streamlit as st

st.title("Hello World!")

st.write("This is my first Streamlit app.")