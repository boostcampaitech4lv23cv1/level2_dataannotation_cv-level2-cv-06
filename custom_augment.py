import random
import albumentations as A
import numpy as np
import random
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import PolygonsOnImage
import cv2
import imgaug as ia


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
                iaa.Rotate((1, 359)),
                iaa.ElasticTransformation(),
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
                    A.RandomBrightnessContrast(p=1),
                    A.ImageCompression(p=1),
                ],
                p=1,
            )
            return transform(image=image)["image"], polygons
        else:
            transform = iaa.imgcorruptlike.Pixelate()
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
        cnt = 0
        while cnt < 1000:
            img, polygon = iaa.CropToFixedSize(length, length)(
                image=image, polygons=polygons
            )

            is_valid = any(int(poly.label) == 1 for poly in polygon)
            if len(polygon) != 0 and is_valid:
                image, polygons = img, polygon
                break
            cnt += 1
        polygons = PolygonsOnImage(polygons, shape=image.shape).remove_out_of_image()
        return image, polygons.polygons

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

        erase_list = []
        for i, (poly, label) in enumerate(zip(polygons, labels)):
            out = (
                any(
                    pts[0] >= w or pts[0] < 0 or pts[1] >= h or pts[1] < 0
                    for pts in poly
                )
                or len(poly) != 4
            )

            if out:
                poly_copy = np.array(poly, dtype=np.int32)
                img = cv2.fillConvexPoly(img, poly_copy, [255, 255, 255])
                erase_list.append(i)

        res_polygons, res_labels = [], []
        for i, (poly, label) in enumerate(zip(polygons, labels)):
            if i not in erase_list:
                res_polygons.append(poly)
                res_labels.append(label)
        return img, res_polygons, res_labels

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
        resize_image, resize_polygons = self.resize(image, polygons)

        re_psoi = ia.PolygonsOnImage(resize_polygons, shape=resize_image.shape)
        re_image_with_polys = re_psoi.draw_on_image(
            resize_image, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0)
        )

        # 2 randomsample
        for transform in transform_list:
            trans_image, trans_polygons = transform(resize_image, resize_polygons)
        trans_psoi = ia.PolygonsOnImage(trans_polygons, shape=trans_image.shape)
        trans_image_with_polys = trans_psoi.draw_on_image(
            trans_image, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0)
        )

        # crop
        crop_image, crop_polygons = self.crop(trans_image, trans_polygons)

        crop_psoi = ia.PolygonsOnImage(crop_polygons, shape=crop_image.shape)
        crop_image_with_polys = crop_psoi.draw_on_image(
            crop_image, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0)
        )

        mask_image, bboxes, labels = self.masking_image(crop_image, crop_polygons)
        # fill
        if bboxes:

            masked_img = ia.PolygonsOnImage(
                [ia.Polygon(bbox) for bbox in bboxes], shape=mask_image.shape
            ).draw_on_image(
                mask_image, alpha_points=0, alpha_face=0.5, color_lines=(255, 0, 0)
            )

        return dict(image=mask_image, bboxes=bboxes, labels=labels)
