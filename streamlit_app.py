from collections import namedtuple
import altair as alt
import math
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import image
from run_eval import ret

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


with st.echo(code_location='below'):
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        img = image.imread(uploaded_file)
        #img = np.load(uploaded_file)

        st.write(ret(img)[0])
