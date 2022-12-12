import streamlit as st

import os
from typing import List
from tools.utils import *
from tools.view import *

st.set_page_config(
    page_title="Image Viewer",
    page_icon="📷",
)

with st.sidebar:
    datasets = get_data_dirs()

    dataset_path = os.path.join(
        DATA_DIR_PATH,
        st.selectbox(
            "Select Dataset", datasets, on_change=change_page_session, args=[0]
        ),
    )
    if "test" == dataset_path:
        ann_path = os.path.join(DATA_DIR_PATH, dataset_path, "ufo/output.json")
    else:
        ann_path = os.path.join(DATA_DIR_PATH, dataset_path, "ufo/train.json")

    df = set_image(ann_path)
    set_page_session()
    group, dataset_path, path = view_img_selector(df, dataset_path)
view_image(group, dataset_path, path)
