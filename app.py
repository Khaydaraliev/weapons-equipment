import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title("Weapons & Equipment")

file = st.file_uploader('Rasm yuklash', type=['png', 'jpg', 'jpeg','gif','svg'])
if file:
    st.image(file)
    img = PILImage.create(file)

    model = load_learner('weapons_equipment_model.pkl')

    pred,  pred_id, probs=model.predict(img)
    st.subheader(f'Bashorat: {pred}')
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig=px.bar(x=probs*100 , y=model.dls.vocab)
    st.plotly_chart(fig)
