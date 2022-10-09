from collections import namedtuple
import altair as alt
import math
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import image
from run_eval import ret

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

