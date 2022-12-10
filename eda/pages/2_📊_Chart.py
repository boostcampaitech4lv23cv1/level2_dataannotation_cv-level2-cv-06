import streamlit as st

import os
from typing import List
from utils import *
from view import *

st.set_page_config(
    page_title="Chart Viewer",
    page_icon="📊",
    layout="wide",
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

    data_dict = load_ann(dataset_path)
    # view_dist_chkbox(data_dict)
    view_dist_multiselect(data_dict)
view_dist(data_dict)
