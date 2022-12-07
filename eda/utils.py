import pandas as pd
import json
import numpy as np
import streamlit as st
from pathlib import Path


def set_image(path):
    """
    path: path of ufo file
    return df with points and image id
    """
    df = pd.DataFrame()
    image_ids = []
    x1, x2, x3, x4, y1, y2, y3, y4 = [], [], [], [], [], [], [], []

    with Path(path).open(encoding="utf8") as f:
        data = json.load(f)

    for image_key, image_value in data["images"].items():
        word_ann = image_value["words"]
        for word in word_ann.values():
            image_ids.append(image_key)
            x1.append(word["points"][0][0])
            y1.append(word["points"][0][1])
            x2.append(word["points"][1][0])
            y2.append(word["points"][1][1])
            x3.append(word["points"][2][0])
            y3.append(word["points"][2][1])
            x4.append(word["points"][3][0])
            y4.append(word["points"][3][1])
    (
        df["image_ids"],
        df["x1"],
        df["x2"],
        df["x3"],
        df["x4"],
        df["y1"],
        df["y2"],
        df["y3"],
        df["y4"],
    ) = (image_ids, x1, x2, x3, x4, y1, y2, y3, y4)

    return df
