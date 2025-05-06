import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('model_nilesh.h5', custom_objects={})

# Constants
vocab_size = 10000
max_length = 100

# Streamlit UI
st.set_page_config(page_title="Misinformation Classifier", layout="centered")

st.title("🚨 Tweet Misinformation Classifier")
st.write("Enter a tweet to check if it contains misinformation.")

tweet_input = st.text_area("Tweet Text", height=150)

if st.button("Classify"):
    if tweet_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocessing
        seq = tokenizer.texts_to_sequences([tweet_input])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

        # Prediction
        prediction = model.predict(padded)[0][0]
        label = "✅ Not Misinformation" if prediction < 0.5 else "⚠️ Misinformation"

        st.markdown(f"### Prediction: {label}")
        st.write(f"**Confidence Score**: {prediction:.4f}")
