import random
import cv2
import albumentations as A
from straug.camera import Contrast, Brightness, JpegCompression
from straug.weather import Fog, Frost, Shadow
from straug.blur import GaussianBlur, MotionBlur, ZoomBlur
from straug.noise import GaussianNoise

# class geometry flip,rotate,perspective


class geometry:
    def __init__(self):
        # self.degree = random.randint(1, 30)
        pass

    """
    def rotate(self, image, bbox):
        h, w, c = image.shape
        cx, cy = w / 2, h / 2
    """
        
    def hflip(self, image, bbox):
        transform = A.HorizontalFlip(p=1)
        length = len(bbox) 
        new_box = np.zeros(length, 4, 2)
        for i in range(length):
            for j in range(4):
                new_box[i][j][0] = weight - box[i][j][0]
        return transform(image), new_box

    
    def vflip(self, image, bbox):
        transform = A.VerticalFlip(p=1)
        length = len(bbox) 
        new_box = np.zeros(length, 4, 2)
        for i in range(length):
            for j in range(4):
                new_box[i][j][1] = height - box[i][j][1]
        return transform(image), new_box
    
    def __call__(self, image, bbox):
        transform = random.sample([hflip, vflip], 1)
        return transform(image, bbox)



class noise:
    def __init__(self):
        self.transform_list = [
            GaussianNoise(),
        ]

    def __call__(self, image):
        transform = random.sample(self.transform_list, 1)
        return transform(image)


class blur:
    def __init__(self):
        self.transform_list = [
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
            A.GlassBlur(p=1),
        ]

    def __call__(self, image):
        transform = random.sample(self.transform_list, 1)
        return transform(image=image)["image"]


class weather:
    def __init__(self):
        self.transform_list = [
            Fog(),
            Frost(),
            Shadow(),
        ]

    def __call__(self, image):
        transform = random.sample(self.transform_list, 1)
        return transform(image)


class camera:
    def __init__(self):
        self.transform_list = A.Compose(
            [
                A.RandomBrightnessContrast(),
                A.JpegCompression(),
            ]
        )

    def __call__(self, image):
        transform = random.sample(self.transform_list, 1)
        return transform(image=image)["image"]


class process:
    def __init__(self):
        self.transform_list = [
            A.Posterize(p=1),
            A.Equalize(p=1),
            A.Solarize(p=1),
            A.InvertImg(p=1),
        ]

    def __call__(self, image):
        transform = random.sample(self.transform_list, 1)
        return transform(image=image)["image"]
    
    
class augment:
    def __init__(self):
        self.geometry = geometry()
        self.blur = blur()
        self.noise = noise()
        self.weather = weather()
        self.camera = camera()
        self.process = process()
        
    def call(self, img, bbox):
        transform_list = random.sample(
            [
                self.geometry,
                self.blur,
                self.noise,
                self.weather,
                self.camera,
                self.process,
                self.hflip,
                self.vflip,
            ],
            2,
        )
        for transform in transform_list:
            img, bbox = transform(img, bbox)

        return dict(image=img, bbox=bbox)
