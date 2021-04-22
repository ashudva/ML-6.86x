from six import string_types
import streamlit as st
import torch
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from torch._C import dtype
from train_utils import Flatten
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Flatten(),
            nn.Linear(720, 128),
            nn.Dropout(0.5)
        )

        self.linear_first_digit = nn.Linear(128, 10)
        self.linear_second_digit = nn.Linear(128, 10)

    def forward(self, x):
        x = self.encoder(x)
        digit1 = self.linear_first_digit(x)
        digit2 = self.linear_second_digit(x)
        return digit1, digit2


model = torch.load('models/MLP_epoch_30.pt')
model.eval()

st.title("Handwritten Multiple Digita Recognition")
st.text(" ")
st.text(" ")

mode = st.checkbox("Draw or Delete", True)
# define size of the canvas
SIZE = 192
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img1 = cv2.resize(canvas_result.image_data.astype('uint8'), (42, 28))
    img2 = cv2.resize(canvas_result.image_data.astype('float'), (42, 28))
    rescaled = cv2.resize(img1, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image for the model')
    st.image(rescaled)

if st.button('Predict the Digit'):
    test_x = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    test_x = test_x.reshape(1, 42, 28)
    test_x = (torch.from_numpy(test_x)).float()
    test_x = (test_x).unsqueeze(0)
    test_x = test_x.to('cuda')
    val1, val2 = model(test_x)
    val1, val2 = val1[0].to('cpu').detach(
    ).numpy(), val2[0].to('cpu').detach().numpy()
    val1 = np.abs(val1)
    val2 = np.abs(val2)
    val1 = val1 / np.sum(val1)
    val2 = val2 / np.sum(val2)
    st.write(f'result: {np.argmax(val1)}')
    st.write(f'result: {np.argmax(val2)}')
    st.bar_chart(val1)
    st.bar_chart(val2)
