from .utils import *


def view_dist_multiselect(data_dict: dict):
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
    return st.multiselect("Select Distribution", dist_list)


def view_dist_chkbox(data_dict: dict):
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
    chkboxes = []
    for dist_name in dist_list:
        chkboxes.append(st.checkbox(dist_name))


def view_dist(data_dict: dict, dist_list: list):
    """
    data_dict: processed annotation json file
    render distributions
    """

    with st.container():
        col1, col2 = st.columns(2)
        col1.header("Testset")
        col2.header("Trainset")

    for dist_name in dist_list:
        with st.container():
            col1, col2 = st.columns(2)
            col1.image(globals()["testset_dist_imshow"](dist_name + ".png"))
            col2.pyplot(globals()[dist_name](data_dict))


def view_img_selector(df: pd.DataFrame, dataset_path: str):

    group = df.groupby("image_ids")
    img_paths = list(group.groups.keys())
    path_lst = [(idx, path) for idx, path in enumerate(img_paths)]
    pages = split_page(path_lst)

    pages
    index, path = st.radio(
        "choose image",
        options=pages[st.session_state.page],
        format_func=lambda x: f"{x[1]}",
    )
    if st.session_state.page == 0:
        prev_flag = True
    else:
        prev_flag = False

    if st.session_state.page == len(pages) - 1:
        next_flag = True
    else:
        next_flag = False

    col1, col2 = st.columns(2)

    with col1:
        st.button(
            "< prev page",
            on_click=change_page_session,
            args=[st.session_state.page - 1],
            disabled=prev_flag,
        )
        # ????????????
        st.button(
            "<< first page",
            on_click=change_page_session,
            args=[0],
            disabled=prev_flag,
        )

    with col2:
        st.button(
            "next page >",
            on_click=change_page_session,
            args=[st.session_state.page + 1],
            disabled=next_flag,
        )
        # ???????????????
        st.button(
            "last page >>",
            on_click=change_page_session,
            args=[len(pages) - 1],
            disabled=next_flag,
        )

    st.text(f"?????? ?????????: {st.session_state.page}")
    page_2_move = st.text_input("page to move")
    try:
        p = int(page_2_move)
        if p >= len(pages):
            st.text(f"????????? ????????? ?????????????????? | ????????? ??????: 0~{len(pages)-1}")
        else:
            st.button("????????? ??????", on_click=change_page_session, args=[p])
    except:
        st.text("????????? ??????????????????")

    return group, dataset_path, path


def view_image(group, dataset_path, path):
    image = draw_image(group, dataset_path, path)
    st.image(image)
