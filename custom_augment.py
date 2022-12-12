import random
import cv2
import albumentations as A
from straug.camera import Contrast, Brightness, JpegCompression
from straug.weather import Fog, Frost, Shadow
from straug.blur import GaussianBlur, MotionBlur, ZoomBlur
from straug.noise import GaussianNoise
from straug.warp import Distort

# class geometry flip,rotate,perspective

"""
class geometry:
    def __init__(self):
        self.degree = random.randint(1, 30)

    def rotate(self, image, bbox):
        h, w, c = image.shape
        cx, cy = w / 2, h / 2
"""


class Warp:
    def __init__(self):
        self.transform_list = [
            Distort(),
        ]

    def __call__(self, image, bbox):
        transform = random.sample(self.transform_list, 1)
        return transform(image)


class Noise:
    def __init__(self):
        self.transform_list = [
            GaussianNoise(),
        ]

    def __call__(self, image, bbox):
        transform = random.sample(self.transform_list, 1)
        return transform(image)


class Blur:
    def __init__(self):
        self.transform_list = [
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
            A.GlassBlur(p=1),
        ]

    def __call__(self, image, bbox):
        transform = random.sample(self.transform_list, 1)
        return transform(image=image)["image"]


class Weather:
    def __init__(self):
        self.transform_list = [
            Fog(),
            Frost(),
            Shadow(),
        ]

    def __call__(self, image, bbox):
        transform = random.sample(self.transform_list, 1)
        return transform(image)


class Camera:
    def __init__(self):
        self.transform_list = [
            A.RandomBrightnessContrast(),
            A.ImageCompression(),
        ]

    def __call__(self, image, bbox):
        transform = random.sample(self.transform_list, 1)
        return transform(image=image)["image"]


class Process:
    def __init__(self):
        self.transform_list = [
            A.Posterize(p=1),
            A.Equalize(p=1),
            A.Solarize(p=1),
            A.InvertImg(p=1),
        ]

    def __call__(self, image, bbox):
        transform = random.sample(self.transform_list, 1)
        return transform(image=image)["image"]


class Augment:
    def __init__(self):
        # self.geometry = geometry()
        self.blur = Blur()
        self.noise = Noise()
        self.weather = Weather()
        self.camera = Camera()
        self.process = Process()

    def __call__(self, img, bbox):
        transform_list = random.sample(
            [
                # self.geometry,
                self.blur,
                self.noise,
                self.weather,
                self.camera,
                self.process,
            ],
            2,
        )
        for transform in transform_list:
            print("$$$$$$$$$$$$$$$$$$")
            print(transform)
            img, bbox = transform(img, bbox)

        return dict(image=img, bbox=bbox)
