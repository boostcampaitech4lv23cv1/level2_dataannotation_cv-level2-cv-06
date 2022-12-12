# from utils import *
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import cv2
import os
import streamlit as st
import pandas as pd

HOME_PATH = os.path.expanduser("~")
DATA_DIR_PATH = os.path.join(HOME_PATH, "input/data")
TESTSET_DIST_PATH = os.path.join("./testset_dist")


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_count_df(
    df: pd.DataFrame,
    field: str,
    random_sample: bool,
    color: str,
    rotation: float,
    xlabel: str,
    ylabel: str,
    title: str,
):
    """ """
    fig, ax = plt.subplots(figsize=(10, 6))
    if random_sample:
        df = df.sample(n=50, random_state=1)
    bars = ax.bar(
        df[field], df[field + "_count"], color=color, align="center", alpha=0.5
    )
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() * (1 / 2),
            b.get_height() + 0.1,
            df.iloc[i][field + "_count"],
            ha="center",
            fontsize=13,
        )
    ax.set_xlabel(xlabel, fontsize=13, rotation=rotation)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    return fig


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_dist(df, field, bins, color, xlabel, ylabel, title):
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.distplot(df[field], bins=bins, color=color, ax=ax)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    return fig


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_dist_list(target_list, bins, color, xlabel, ylabel, title):
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.distplot(target_list, bins=bins, color=color, ax=ax)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    return fig


@st.cache
def testset_dist_imshow(filename):
    img = cv2.imread(os.path.join(TESTSET_DIST_PATH, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    img_height = 1000
    img = cv2.resize(img, (img_height, int(img_height * img_shape[0] / img_shape[1])))
    return img
