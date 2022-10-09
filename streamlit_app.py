from collections import namedtuple
import altair as alt
import math
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import image
from run_eval import ret
from model_unet import HackathonModel

"""
# Welcome to Silos Detect by Brasil'IA

## Classification problem:
"""

uploaded_file = st.file_uploader("Choose a file")
    
if uploaded_file is not None:
    img = image.imread(uploaded_file)
    #img = np.load(uploaded_file)

    st.write(ret(img)[0])


"""
## Segmentation problem:
"""
    input = defaultdict()
    input['img'] = img[None, :]
    model = HackathonModel.load_from_checkpoint("model_weights/unet.ckpt")
    segmented = model(input)
    #img = np.load(uploaded_file)
    fig = plt.figure()
    plt.imshow(segmented.cpu().detach().numpy(), cmap='Greys')
    st.write(fig)

    st.write(ret(img)[0])