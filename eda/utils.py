import pandas as pd
import json
import numpy as np
import streamlit as st
from pathlib import Path
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from stqdm import stqdm

HOME_PATH = os.path.expanduser("~")
DATA_DIR_PATH = os.path.join(HOME_PATH, "input/data")
TESTSET_DIST_PATH = os.path.join("./testset_dist")


def get_data_dirs():
    """
    return data directory list
    """
    return [i for i in os.listdir(DATA_DIR_PATH) if not i.startswith(".")]


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


def read_json(filename):
    with Path(filename).open(encoding="utf8") as handle:
        ann = json.load(handle)
    return ann


def dataset_selectbox(key):
    return st.selectbox("Dataset", get_data_dirs(), key=key)


def gallery_contents(dataset_dir):
    data = load_ann(dataset_dir)
    img_dir_path = os.path.join(DATA_DIR_PATH, dataset_dir, "images")
    img_list = os.listdir(img_dir_path)
    img_list = sorted([i for i in os.listdir(img_dir_path) if not i.startswith(".")])
    img_name = st.selectbox("Image", img_list)
    img_path = os.path.join(img_dir_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for _, word_value in data["images"][img_name]["words"].items():
        pts = np.array(word_value["points"]).astype(int)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], True, (255, 0, 0), 3)

    st.image(img, width=600)


def get_box_size(quads):
    """단어 영역의 사각형 좌표가 주어졌을 때 가로, 세로길이를 계산해주는 함수.
    TODO: 각 변의 길이를 단순히 max로 처리하기때문에 직사각형에 가까운 형태가 아니면 약간 왜곡이 있다.
    Args:
        quads: np.ndarray(n, 4, 2) n개 단어 bounding-box의 4개 점 좌표 (단위 pixel)
    Return:
        sizes: np.ndarray(n, 2) n개 box의 (height, width)쌍
    """
    dists = []
    for i, j in [
        (1, 2),
        (3, 0),
        (0, 1),
        (2, 3),
    ]:  # [right(height), left(height), upper(width), lower(width)] sides
        dists.append(np.linalg.norm(quads[:, i] - quads[:, j], ord=2, axis=1))

    dists = np.stack(dists, axis=-1).reshape(
        -1, 2, 2
    )  # shape (n, 2, 2) widths, heights into separate dim
    return np.rint(dists.mean(axis=-1)).astype(int)


def rectify_poly(poly, direction, img_w, img_h):
    """일반 polygon형태인 라벨을 크롭하고 rectify해주는 함수.
    Args:
        poly: np.ndarray(2n+4, 2) (where n>0), 4, 6, 8
        image: np.ndarray opencv 포멧의 이미지
        direction: 글자의 읽는 방향과 진행 방향의 수평(Horizontal) 혹은 수직(Vertical) 여부
    Return:
        rectified: np.ndarray(2, ?) rectify된 단어 bbox의 사이즈.
    """

    n_pts = poly.shape[0]
    assert n_pts % 2 == 0
    if n_pts == 4:
        size = get_box_size(poly[None])
        h = size[:, 0] / img_h
        w = size[:, 1] / img_w
        return np.stack((h, w))

    def unroll(indices):
        return list(zip(indices[:-1], indices[1:]))

    # polygon하나를 인접한 사각형 여러개로 쪼갠다.
    indices = list(range(n_pts))
    if direction == "Horizontal":
        upper_pts = unroll(indices[: n_pts // 2])  # (0, 1), (1, 2), ... (4, 5)
        lower_pts = unroll(indices[n_pts // 2 :])[::-1]  # (8, 9), (7, 8), ... (6, 7)

        quads = np.stack(
            [poly[[i, j, k, l]] for (i, j), (k, l) in zip(upper_pts, lower_pts)]
        )
    else:
        right_pts = unroll(indices[1 : n_pts // 2 + 1])  # (1, 2), (2, 3), ... (4, 5)
        left_pts = unroll(
            [0] + indices[: n_pts // 2 : -1]
        )  # (0, 9), (9, 8), ... (7, 6)

        quads = np.stack(
            [poly[[i, j, k, l]] for (j, k), (i, l) in zip(right_pts, left_pts)]
        )

    sizes = get_box_size(quads)
    if direction == "Horizontal":
        h = sizes[:, 0].max() / img_h
        widths = sizes[:, 1]
        w = np.sum(widths) / img_w
        return np.stack((h, w)).reshape(2, -1)
    elif direction == "Vertical":
        heights = sizes[:, 0]
        w = sizes[:, 1].max() / img_w
        h = np.sum(heights) / img_h
        return np.stack((h, w)).reshape(2, -1)
    else:
        h = sizes[:, 0] / img_h
        w = sizes[:, 1] / img_w
        return np.stack((h, w), -1)


def load_ann(ann_path):

    data = read_json(ann_path)

    df = {}
    df["image"] = []
    df["word_counts"] = []
    df["image_width"] = []
    df["image_height"] = []
    df["image_tags"] = []

    seq_length = []
    hor_sizes = []
    ver_sizes = []
    irr_sizes = []
    languages = []
    orientation = []
    word_tags = []

    for image_key, image_value in stqdm(data["images"].items()):
        df["image"].append(image_key)
        img_w = image_value["img_w"]
        img_h = image_value["img_h"]
        df["image_width"].append(img_w)
        df["image_height"].append(img_h)
        df["image_tags"].append(image_value["tags"])
        df["image_tags"] = [
            ["None"] if v is None else v for v in df["image_tags"]
        ]  # our data does not inlcude multi-tag images
        word_ann = image_value["words"]
        count_ill = 0
        for word in word_ann.values():
            if "word_tags" in word:
                wt = "word_tags"
            elif "tags" in word:
                wt = "tags"
            else:
                print("what?")
            if word["illegibility"] == False:
                orientation.append(word["orientation"])
                orientation = [v for v in orientation]
                seq_length.append(len(word["transcription"]))
                languages.append(word["language"])
                languages = [
                    ["None"] if v is None else v for v in languages
                ]  # our data does not inlcude multi-language words

                if word[wt] != None:
                    word_tags.extend(word[wt][:])
                elif word[wt] == None:
                    word_tags.append("None")
                poly = np.int32(word["points"])
                size = rectify_poly(poly, word["orientation"], img_w, img_h)
                if word["orientation"] == "Horizontal":
                    hor_sizes.append(size)
                elif word["orientation"] == "Vertical":
                    ver_sizes.append(size)
                else:
                    irr_sizes.append(size)

            else:
                count_ill += 1
        df["word_counts"].append(len(word_ann) - count_ill)

    all_sizes = hor_sizes + ver_sizes + irr_sizes
    quad_area = [all_sizes[i][0] * all_sizes[i][1] for i in range(len(all_sizes))]
    total_area = []
    for s in quad_area:
        if s.shape[0] == 1:
            total_area.append(np.sum(s[0]))
        else:
            total_area.append(np.sum(s))

    hor_aspect_ratio = [
        hor_sizes[i][1] / hor_sizes[i][0] for i in range(len(hor_sizes))
    ]
    ver_aspect_ratio = [
        ver_sizes[i][1] / ver_sizes[i][0] for i in range(len(ver_sizes))
    ]
    word_df = {}
    word_df["index"] = [i for i in range(len(total_area))]
    word_df["orientation"] = orientation
    word_df["language"] = languages
    word_df["bbox_size"] = total_area
    word_df = pd.DataFrame.from_dict(word_df)
    word_df["language"] = word_df["language"].apply(lambda x: ",".join(map(str, x)))

    image_df = pd.DataFrame.from_dict(df)
    image_df["image_tags"] = image_df["image_tags"].apply(
        lambda x: ",".join(map(str, x))
    )

    data_dict = {}
    data_dict["image_df"] = image_df
    data_dict["word_tags"] = word_tags
    data_dict["word_df"] = word_df
    data_dict["hor_aspect_ratio"] = hor_aspect_ratio
    data_dict["ver_aspect_ratio"] = ver_aspect_ratio
    return data_dict


def image_size_dist(data_dict):
    image_df = data_dict["image_df"]
    g = sns.jointplot(
        data=image_df, x="image_width", y="image_height", kind="kde", space=0, color="r"
    )
    g.set_axis_labels("Image Width", "Image Height")
    return g


def image_tag_dist(data_dict):
    image_df = data_dict["image_df"]
    img_tag_df = create_count_df(df=image_df, field="image_tags", index="image")
    return plot_count_df(
        df=img_tag_df,
        field="image_tags",
        random_sample=False,
        color="g",
        rotation=0,
        xlabel="image tag",
        ylabel="Number of image tag",
        title="Image Tag Distribution",
    )


def word_tag_dist(data_dict):
    word_tags = data_dict["word_tags"]
    word_tag_df = pd.DataFrame(word_tags, columns=["word_tags"])
    word_tag_df["index"] = [i for i in range(len(word_tags))]
    word_tag_df = create_count_df(word_tag_df, field="word_tags", index="index")
    return plot_count_df(
        df=word_tag_df,
        field="word_tags",
        random_sample=False,
        color="g",
        rotation=0,
        xlabel="word tags",
        ylabel="Count of each word tag",
        title="Word tag Distribution",
    )


def word_per_image_dist(data_dict):
    image_df = data_dict["image_df"]
    return plot_dist(
        df=image_df,
        field="word_counts",
        bins=50,
        color="b",
        xlabel="number of words per Image",
        ylabel="Frequency",
        title="Words per Image Distribution",
    )


def orientation_dist(data_dict):
    word_df = data_dict["word_df"]
    orientation = create_count_df(df=word_df, field="orientation", index="index")
    return plot_count_df(
        df=orientation,
        field="orientation",
        random_sample=False,
        color="g",
        rotation=0,
        xlabel="orientation",
        ylabel="Count of each orientation",
        title="Orientation Distribution",
    )


def language_dist(data_dict):
    word_df = data_dict["word_df"]
    lang = create_count_df(df=word_df, field="language", index="index")
    return plot_count_df(
        df=lang,
        field="language",
        random_sample=False,
        color="g",
        rotation=0,
        xlabel="language",
        ylabel="Count of each language",
        title="Language Distribution",
    )


def bbox_size_dist(data_dict):
    word_df = data_dict["word_df"]
    return plot_dist(
        df=word_df,
        field="bbox_size",
        bins=200,
        color="r",
        xlabel="BBOX size",
        ylabel="Frequency",
        title="BBOX size",
    )


def hor_aspect_ratio_dist(data_dict):
    hor_aspect_ratio = data_dict["hor_aspect_ratio"]
    return plot_dist_list(
        hor_aspect_ratio,
        bins=20,
        color="r",
        xlabel="Aspect Ratio (BBOX Width / BBOX Height)",
        ylabel="Frequency",
        title="Aspect Ratio Distribution (Horizontal words)",
    )


def ver_aspect_ratio_dist(data_dict):
    ver_aspect_ratio = data_dict["ver_aspect_ratio"]
    return plot_dist_list(
        ver_aspect_ratio,
        bins=20,
        color="r",
        xlabel="Aspect Ratio (BBOX Width / BBOX Height)",
        ylabel="Frequency",
        title="Aspect Ratio Distribution (Veritcal words)",
    )


def create_count_df(df, field, index):
    count = df.groupby(field)[index].count().sort_values(ascending=False)
    count_df = count.to_frame().reset_index()
    count_df.columns = [field, field + "_count"]
    return count_df


def plot_count_df(df, field, random_sample, color, rotation, xlabel, ylabel, title):
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
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    return fig


def plot_dist(df, field, bins, color, xlabel, ylabel, title):
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.distplot(df[field], bins=bins, color=color, ax=ax)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    return fig


def plot_dist_list(target_list, bins, color, xlabel, ylabel, title):
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.distplot(target_list, bins=bins, color=color, ax=ax)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    return fig


def testset_dist_imshow(filename):
    img = cv2.imread(os.path.join(TESTSET_DIST_PATH, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    img_height = 1000
    img = cv2.resize(img, (img_height, int(img_height * img_shape[0] / img_shape[1])))
    return img


def view_dist(data_dict: dict):
    """
    data_dict: processed annotation json file
    render distributions
    """
    dist_list = [
        "image_size_dist",
        "image_tag_dist",
        "word_tag_dist",
        "orientation_dist",
        "language_dist",
        "bbox_size_dist",
        "hor_aspect_ratio_dist",
        "ver_aspect_ratio_dist",
    ]

    with st.container():
        col1, col2 = st.columns(2)
        col1.header("Testset")
        col2.header("Trainset")

    for dist_name in dist_list:
        with st.container():
            col1, col2 = st.columns(2)
            col1.image(globals()["testset_dist_imshow"](dist_name + ".png"))
            col2.pyplot(globals()[dist_name](data_dict))


def draw_image(group, dataset_path: str, img_path: str):
    """
    group: grouped df by image id
    img_path: image folder path
    return cv2 image with annotation
    """
    image = cv2.imread(os.path.join(DATA_DIR_PATH, dataset_path, "images", img_path))
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
        # x_min = min(bbox.x1, bbox.x2, bbox.x3, bbox.x4)
        # y_min = min(bbox.y1, bbox.y2, bbox.y3, bbox.y4)
        image = cv2.polylines(image, [pts], True, [0, 0, 0], thickness=3)
    return image


def view_image(df: pd.DataFrame, dataset_path: str):
    """ """

    group = df.groupby("image_ids")
    img_paths = list(group.groups.keys())
    path_lst = [(idx, path) for idx, path in enumerate(img_paths)]

    col1, col2, col3 = st.columns([1, 8, 1])

    with col2:
        # 10개 단위로 페이지 쪼개기
        pages = split_page(path_lst)
        index, path = st.radio(
            "choose image",
            options=pages[st.session_state.page],
            format_func=lambda x: f"{x[1]}",
        )

        st.text(f"현재 페이지: {st.session_state.page}")

        page_2_move = st.text_input("page to move")

        try:
            p = int(page_2_move)
            if p >= len(pages):
                st.text(f"페이지 범위를 벗어났습니다 | 페이지 범위: 0~{len(pages)-1}")
            else:
                st.button("페이지 이동", on_click=change_page, args=[p])
        except:
            st.text("숫자를 입력해주세요")

    with col1:
        if st.session_state.page == 0:
            prev_flag = True
        else:
            prev_flag = False
        # 처음으로
        st.button(
            "<< first page",
            on_click=change_page,
            args=[0],
            disabled=prev_flag,
        )
        st.button(
            "< prev page",
            on_click=change_page,
            args=[st.session_state.page - 1],
            disabled=prev_flag,
        )

    with col3:
        if st.session_state.page == len(pages) - 1:
            next_flag = True
        else:
            next_flag = False
        # 마지막으로
        st.button(
            "last page >>",
            on_click=change_page,
            args=[len(pages) - 1],
            disabled=next_flag,
        )
        st.button(
            "next page >",
            on_click=change_page,
            args=[st.session_state.page + 1],
            disabled=next_flag,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(" ")

    with c2:
        image = draw_image(group, dataset_path, path)
        st.image(image)

    with c3:
        st.write(" ")


def set_session():
    if "page" not in st.session_state:
        st.session_state.page = 0


@st.cache()
def split_page(path_lst: list):
    page_lst = []
    tmp = []
    cnt = 0
    for ele in path_lst:
        if cnt > 9:
            page_lst.append(tmp)
            cnt = 0
            tmp = []
        tmp.append(ele)
        cnt += 1
    return page_lst


def change_page(page_num: int):
    st.session_state.page = page_num
