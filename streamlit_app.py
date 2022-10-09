from collections import namedtuple, defaultdict
import altair as alt
import math
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import image
from run_eval import ret
from model_unet import HackathonModel
import torch
import matplotlib.pyplot as plt

"""
# Welcome to Silos Detect by Brasil'IA

## Classification problem:
"""

uploaded_file = st.file_uploader("Choose a file")
    
if uploaded_file is not None:
    img = image.imread(uploaded_file)

    st.write("Silos presence probability {:.2f}".format(ret(img)[0].item()))

    input = defaultdict()
    input['img'] = torch.tensor(img[None, :])
    model = HackathonModel.load_from_checkpoint("model_weights/unet.ckpt")
    segmented = model(input)
    fig = plt.figure()
    plt.imshow(segmented.cpu().detach().numpy(), cmap='Greys')
    st.write(fig)