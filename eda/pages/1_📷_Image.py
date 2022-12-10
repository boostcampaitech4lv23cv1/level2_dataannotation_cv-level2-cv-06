import streamlit as st

import os
from typing import List
from utils import *
from view import *

st.set_page_config(
    page_title="Image Viewer",
    page_icon="📷",
    # layout="wide",
)

with st.sidebar:
    datasets = get_data_dirs()

    dataset_path = os.path.join(
        DATA_DIR_PATH,
        st.selectbox("Dataset Selection", datasets, on_change=change_page, args=[0]),
    )
    if "test" == dataset_path:
        ann_path = os.path.join(DATA_DIR_PATH, dataset_path, "ufo/output.json")
    else:
        ann_path = os.path.join(DATA_DIR_PATH, dataset_path, "ufo/train.json")

    df = set_image(ann_path)
    set_session()
    group, dataset_path, path = view_img_selector(df, dataset_path)

view_image(group, dataset_path, path)
