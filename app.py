import streamlit as st
from ultralytics import YOLO
from PIL import Image
st.title("🕸 Spider-Verse Character Guess")
model = YOLO("best.pt")
uploaded = st.file_uploader("Upload image",
type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image")
    results = model(image)[0]
    name = results.names[results.probs.top1]
    conf = results.probs.top1conf.item() * 100
    st.success(f"Prediction: {name}")
    st.write(f"Confidence: {conf:.2f}%")
