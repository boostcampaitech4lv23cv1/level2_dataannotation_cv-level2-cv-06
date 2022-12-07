import streamlit as st
import cv2
import os
import pandas as pd
from typing import List
import sys
from utils import *


def draw_image(group, img_path):
    """
    group: grouped df by image id
    img_path: image folder path
    return cv2 image with annotation
    """
    image = cv2.imread(os.path.join("../../input/data", data, "images", img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = group.get_group(img_path)

    for _, bbox in bboxes.iterrows():
        pts = np.array(
            [
                [bbox.x1, bbox.y1],
                [bbox.x2, bbox.y2],
                [bbox.x3, bbox.y3],
                [bbox.x4, bbox.y4],
            ],
            np.int32,
        )
        x_min = min(bbox.x1, bbox.x2, bbox.x3, bbox.x4)
        y_min = min(bbox.y1, bbox.y2, bbox.y3, bbox.y4)
        image = cv2.polylines(image, [pts], True, [0, 0, 0])
    return image


def view_image(df: pd.DataFrame):
    """
    selectbox: select image and show
    next button: show image by session count

    session count
    selectbox: after select item make session count as image number (for next button)

    session key
    To store prev image/ select image after nextbutton image is not updated without key
    """
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    group = df.groupby("image_ids")
    img_paths = group.groups.keys()
    img_path = st.selectbox("choose image", img_paths)
    if st.session_state.counter == 0 or st.session_state["key"] != str(img_path):
        st.session_state.key = img_path
        image = draw_image(group, img_path)
        show = st.image(image, width=700)
        paths = [*img_paths]
        st.session_state.counter = paths.index(img_path)

    def show_photo(photo):
        """
        if select box has value clear select box image after next button
        """
        if st.session_state.counter == 1:
            show.empty()
        paths = [*img_paths]
        photo = paths[photo]
        image = draw_image(group, photo)
        st.image(image, width=700)

    st.session_state.counter += 1
    photo = st.session_state.counter
    next = st.button("next", on_click=show_photo, args=([photo]))


st.set_page_config(layout="wide")
st.title("Data visualization")


dataset = (
    "ICDAR17_Korean",
    "ICDAR2015",
    "test",
    "ICDAR17",
)  # , "ICDAR17_Korean_test")  select list of dataset should be folder name of dataset including ufo and images
data = st.selectbox("Dataset Selection", dataset)
path = os.path.join("../../input/data", data, "ufo/train.json")

if (
    "test" == data
):  # validation set with crawling data (made because of difference with file name(output.json/train.json))
    path = os.path.join("../../input/data", data, "ufo/output.json")
"""
#view trainset annotation after validation (ICDAR17_Korean_test)
elif "test" in data: # validation set wi
    data = data.split("_test")[0]
    path = os.path.join("../../input/data", data, "ufo/output.json")
"""

df = set_image(path)
(vz_tab, tmp2) = st.tabs(
    [
        "tmp1",
        "tmp2",
    ]
)
with vz_tab:
    view_image(df)


# 실행 명령어 streamlit run main.py  --server.fileWatcherType none --server.port 30002
