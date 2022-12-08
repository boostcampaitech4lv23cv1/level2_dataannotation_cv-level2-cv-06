import json
import os
import os.path as osp
from glob import glob
from PIL import Image

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset


SRC_DATASET_DIR = "../input/data/"

NUM_WORKERS = 4

IMAGE_EXTENSIONS = {".gif", ".jpg", ".png"}

LANGUAGE_MAP = {
    "Korean": "ko",
    "Latin": "en",
    "Symbols": None,
    "Arabic": "ar",
    "Chinese": "ch",
    "Japanese": "ja",
    "Bangla": "ba",
    "Hindi": "hi",
}
LANGUAGE_IDX = {
    "Arabic": 0,
    "English": 1,
    "French": 2,
    "Chinese": 3,
    "German": 4,
    "Korean": 5,
    "Japanese": 6,
    "Italian": 7,
    "Bangla": 8,
    "Hindi": 9,
}


def get_language_token(x):
    return LANGUAGE_MAP.get(x, "others")


def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)


class MLT19Dataset(Dataset):
    def __init__(self, image_dir1, image_dir2, label_dir, lang, copy_images_to=None):
        image_paths = {
            x
            for x in glob(osp.join(image_dir1, "*"))
            if osp.splitext(x)[1] in IMAGE_EXTENSIONS
        }
        image_paths.update(
            {
                x
                for x in glob(osp.join(image_dir2, "*"))
                if osp.splitext(x)[1] in IMAGE_EXTENSIONS
            }
        )
        label_paths = set(glob(osp.join(label_dir, "*.txt")))
        assert len(image_paths) == len(label_paths)

        sample_ids, samples_info = [], {}
        for image_path in sorted(image_paths):
            sample_id = osp.splitext(osp.basename(image_path))[0]

            label_path = osp.join(label_dir, f"{sample_id}.txt")
            assert label_path in label_paths

            words_info, _ = self.parse_label_file(label_path)
            if int(sample_id.split("_")[-1]) not in range(
                1000 * LANGUAGE_IDX[lang] + 1, 1000 + 1000 * LANGUAGE_IDX[lang] + 1
            ):
                continue
            sample_ids.append(sample_id)
            samples_info[sample_id] = dict(
                image_path=image_path, label_path=label_path, words_info=words_info
            )

        self.sample_ids, self.samples_info = sample_ids, samples_info

        self.copy_images_to = copy_images_to

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_info = self.samples_info[self.sample_ids[idx]]

        image_fname = osp.basename(sample_info["image_path"])
        image = Image.open(sample_info["image_path"])
        img_w, img_h = image.size

        if self.copy_images_to:
            maybe_mkdir(self.copy_images_to)
            image.save(
                osp.join(self.copy_images_to, osp.basename(sample_info["image_path"]))
            )

        license_tag = dict(
            usability=True, public=True, commercial=True, type="CC-BY-SA", holder=None
        )
        sample_info_ufo = dict(
            img_h=img_h,
            img_w=img_w,
            words=sample_info["words_info"],
            tags=None,
            license_tag=license_tag,
        )

        return image_fname, sample_info_ufo

    def parse_label_file(self, label_path):
        def rearrange_points(points):
            start_idx = np.argmin([np.linalg.norm(p, ord=1) for p in points])
            if start_idx != 0:
                points = np.roll(points, -start_idx, axis=0).tolist()
            return points

        with open(label_path, encoding="utf-8") as f:
            lines = f.readlines()

        words_info, languages = {}, set()
        for word_idx, line in enumerate(lines):
            items = line.strip().split(",", 9)
            language, transcription = items[8], items[9]
            points = np.array(items[:8], dtype=np.float32).reshape(4, 2).tolist()
            points = rearrange_points(points)

            illegibility = transcription == "###"
            orientation = "Horizontal"
            language = get_language_token(language)
            words_info[word_idx] = dict(
                points=points,
                transcription=transcription,
                language=[language],
                illegibility=illegibility,
                orientation=orientation,
                word_tags=None,
            )
            languages.add(language)

        return words_info, dict(languages=languages)


def main(lang):
    DST_DATASET_DIR = f"../input/data/ICDAR19_{lang}"

    dst_image_dir = osp.join(DST_DATASET_DIR, "images")

    mlt = MLT19Dataset(
        osp.join(SRC_DATASET_DIR, "ICDAR19_1"),
        osp.join(SRC_DATASET_DIR, "ICDAR19_2"),
        osp.join(SRC_DATASET_DIR, "ICDAR19_GT"),
        lang,
        copy_images_to=dst_image_dir,
    )

    anno = dict(images={})
    with tqdm(total=len(mlt)) as pbar:
        for batch in DataLoader(mlt, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno["images"][image_fname] = sample_info
            pbar.update(1)

    ufo_dir = osp.join(DST_DATASET_DIR, "ufo")
    maybe_mkdir(ufo_dir)
    with open(osp.join(ufo_dir, "train.json"), "w") as f:
        json.dump(anno, f, indent=4)


if __name__ == "__main__":
    for i in LANGUAGE_IDX.keys():
        main(i)
