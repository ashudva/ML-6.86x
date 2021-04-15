from six import string_types
import streamlit as st
import torch
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from torch._C import dtype
import torch.nn.functional as F

model = torch.load('models/CNN10.pt')
model.eval()

st.title("Handwritten Digit Recognition Web App")
st.text(" ")
st.text(" ")
mode = st.checkbox("Draw or Delete", True)
# define size of the canvas
SIZE = 192
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img1 = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img2 = cv2.resize(canvas_result.image_data.astype('float'), (28, 28))
    rescaled = cv2.resize(img1, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image for the model')
    st.image(rescaled)

if st.button('Predict the Digit'):
    test_x = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    test_x = test_x.reshape(1, 28, 28)
    test_x = (torch.from_numpy(test_x)).float()
    test_x = (test_x).unsqueeze(0)
    test_x = test_x.to('cuda')
    val = model(test_x)
    val = val[0].to('cpu').detach().numpy()
    val = np.abs(val)
    val = val / np.sum(val)
    st.write(f'result: {np.argmax(val)}')
    st.bar_chart(val)
    st.text(val)
