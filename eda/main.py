import os
from typing import List
from utils import *
from view import *


st.set_page_config(layout="wide")
st.title("Data visualization")

# select list of dataset should be folder name of dataset including ufo and images
datasets = get_data_dirs()
dataset_path = os.path.join(DATA_DIR_PATH, st.selectbox("Dataset Selection", datasets))

# validation set with crawling data (made because of difference with file name(output.json/train.json))
if "test" == dataset_path:
    ann_path = os.path.join(DATA_DIR_PATH, dataset_path, "ufo/output.json")
else:
    ann_path = os.path.join(DATA_DIR_PATH, dataset_path, "ufo/train.json")
# """
# view trainset annotation after validation (ICDAR17_Korean_test)
# elif "test" in data: # validation set wi
#     data = data.split("_test")[0]
#     path = os.path.join("../../input/data", data, "ufo/output.json")
# """

set_session()
df = set_image(ann_path)

st.write(
    "<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>",
    unsafe_allow_html=True,
)

st.write(
    "<style>div.st-bf{flex-direction:column;} div.st-ag{padding-left:2px;}</style>",
    unsafe_allow_html=True,
)

(vz_tab, dist_tab) = st.tabs(
    [
        "Image Viewer",
        "Data Distribution",
    ]
)

with vz_tab:
    view_image(df, dataset_path)
with dist_tab:
    data_dict = load_ann(dataset_path)
    view_dist(data_dict)


# 실행 명령어 streamlit run main.py  --server.fileWatcherType none --server.port 30002
