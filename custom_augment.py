import random
import albumentations as A
import numpy as np
import random
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import PolygonsOnImage
import cv2


class Warp:
    def __init__(self):
        self.transform = iaa.Affine()

    def __call__(self, image, polygons):
        img, polygons = self.transform(image=image, polygons=polygons)
        return img, polygons


class Geometry:
    def __init__(self):
        self.transform = iaa.OneOf(
            [
                iaa.Fliplr(),
                iaa.Flipud(),
                iaa.Sequential([iaa.Fliplr(), iaa.Flipud()]),
                iaa.PerspectiveTransform(),
                iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # distort
                iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1))),  # curve
                iaa.Rotate((1, 359)),
            ]
        )

    def __call__(self, image, polygons):
        img, polygons = self.transform(image=image, polygons=polygons)
        return img, polygons


class Noise:
    def __init__(self):
        self.transform = iaa.OneOf(
            [
                iaa.imgcorruptlike.GaussianNoise(),
                iaa.imgcorruptlike.ShotNoise(),
                iaa.imgcorruptlike.SpeckleNoise(),
                iaa.imgcorruptlike.ImpulseNoise(),
            ],
        )

    def __call__(self, image, polygons):
        return self.transform(image=image, polygons=polygons)


class Blur:
    def __init__(self):
        self.transform = A.OneOf(
            [
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
                A.GlassBlur(p=1, max_delta=1, iterations=1),
                A.ZoomBlur(p=1),
            ],
            p=1,
        )

    def __call__(self, image, polygons):
        return self.transform(image=image)["image"], polygons


class Weather:
    def __init__(self):
        self.transform = A.OneOf(
            [
                A.RandomRain(p=1, blur_value=1),
                A.RandomSnow(p=1),
                A.RandomFog(p=1),
                A.RandomShadow(p=1),
            ],
            p=1,
        )

    def __call__(self, image, polygons):
        return self.transform(image=image)["image"], polygons


class Camera:
    def __init__(self):
        pass

    def __call__(self, image, polygons):
        if np.random.uniform(0, 1) > 0.5:
            transform = A.OneOf(
                [
                    A.RandomContrast(p=1),
                    A.RandomBrightness(p=1),
                    A.JpegCompression(p=1),
                ],
                p=1,
            )
            return transform(image=image)["image"], polygons
        else:
            transform = iaa.imgcorrupt.Pixelate()
            return transform(image=image, polygons=polygons)


class Process:
    def __init__(self):
        pass

    def __call__(self, image, polygons):
        if np.random.uniform(0, 1) > 0.5:
            transform = A.OneOf(
                [
                    A.Posterize(p=1),
                    A.Solarize(p=1),
                    A.InvertImg(p=1),
                    A.Equalize(p=1),
                    A.ColorJitter(p=1),
                ],
                p=1,
            )
            return transform(image=image)["image"], polygons
        else:
            transform = iaa.pillike.EnhanceSharpness()

            return transform(image=image, polygons=polygons)


class Augment:
    def __init__(self, img_size, crop_size):
        self.img_size = img_size
        self.crop_size = crop_size
        self.geometry = Geometry()
        self.blur = Blur()
        self.noise = Noise()
        self.weather = Weather()
        self.camera = Camera()
        self.process = Process()
        self.warp = Warp()

    def resize(self, image: np.array, polygons):
        h, w, _ = image.shape
        size = self.img_size
        if w > h:
            image, polygons = iaa.Resize(
                {"height": "keep-aspect-ratio", "width": size}
            )(image=image, polygons=polygons)
        else:
            image, polygons = iaa.Resize(
                {"height": size, "width": "keep-aspect-ratio"}
            )(image=image, polygons=polygons)
        return image, polygons

    def crop(self, image: np.array, polygons):
        length = self.crop_size
        h, w, _ = image.shape
        if h >= w and w < length:
            image, polygons = iaa.Resize(
                {"height": int(h * length / w), "width": length}
            )(image=image, polygons=polygons)
        elif h < w and h < length:
            image, polygons = iaa.Resize(
                {"height": length, "width": int(w * length / h)}
            )(image=image, polygons=polygons)

        image, polygons = iaa.CropToFixedSize(length, length)(
            image=image, polygons=polygons
        )
        polygons = PolygonsOnImage(polygons, shape=image.shape).remove_out_of_image()
        return image, polygons

    def poly_to_list(self, polygons):
        # return list type
        annotations, labels = [], []
        for poly in polygons:
            labels.append(poly.label)
            anno = [pts.tolist() for pts in poly]
            annotations.append(anno)
        return annotations, labels

    def masking_image(self, img, polygons):
        polygons, labels = self.poly_to_list(polygons)
        h, w, _ = img.shape
        for (poly, label) in zip(polygons[:], labels[:]):
            out = any(
                pts[0] >= w or pts[0] < 0 or pts[1] >= h or pts[1] < 0 for pts in poly
            )

            if out:
                poly_copy = np.array(poly, dtype=np.int32)
                img = cv2.fillConvexPoly(img, poly_copy, [255, 255, 255])
                polygons.remove(poly)
                labels.remove(label)
        return img, polygons, labels

    def __call__(self, image, polygons):
        transform_list = random.sample(
            [
                self.geometry,
                self.blur,
                self.noise,
                self.weather,
                self.camera,
                self.process,
                self.warp,
            ],
            2,
        )

        # resize
        image, polygons = self.resize(image, polygons)

        for transform in transform_list:
            if transform in [self.geometry, self.warp]:
                image, polygons = transform(image, polygons)
            else:
                image = transform(image)

        # 2 randomsample
        for transform in transform_list:
            image = transform(image)
        # crop
        image, polygons = self.crop(image, polygons)
        image, bboxes, labels = self.masking_image(image, polygons)
        # fill

        return dict(image=image, bboxes=bboxes, labels=labels)
