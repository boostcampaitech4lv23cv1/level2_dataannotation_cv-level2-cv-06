import random
import cv2
import albumentations as A
import numpy as np

# class geometry flip,rotate,perspective


from typing import Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class Rotate:
    def warpAffine(self, src, M, dsize, from_bounding_box_only=False):
        return cv2.warpAffine(src, M, dsize)

    def rotate_image(self, image):
        # get dims, find center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        angle = self.angle
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = self.warpAffine(image, M, (nW, nH), False)

        # image = cv2.resize(image, (w,h))

        return image

    def rotate_point(self, origin, point):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        :param angle: <float> Angle in radians.
            Positive angle is counterclockwise.
        """
        ox, oy = origin
        px, py = point
        angle = self.angle
        angle = math.radians(angle)

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def crop_to_center(self, old_img, new_img):
        """
        Crops `new_img` to `old_img` dimensions
        :param old_img: <numpy.ndarray> or <tuple> dimensions
        :param new_img: <numpy.ndarray>
        :return: <numpy.ndarray> new image cropped to old image dimensions
        """

        if isinstance(old_img, tuple):
            original_shape = old_img
        else:
            original_shape = old_img.shape
        original_width = original_shape[1]
        original_height = original_shape[0]
        original_center_x = original_shape[1] / 2
        original_center_y = original_shape[0] / 2

        new_width = new_img.shape[1]
        new_height = new_img.shape[0]
        new_center_x = new_img.shape[1] / 2
        new_center_y = new_img.shape[0] / 2

        new_left_x = int(max(new_center_x - original_width / 2, 0))
        new_right_x = int(min(new_center_x + original_width / 2, new_width))
        new_top_y = int(max(new_center_y - original_height / 2, 0))
        new_bottom_y = int(min(new_center_y + original_height / 2, new_height))

        # create new img canvas
        # canvas = np.zeros(original_shape)
        canvas = np.full(original_shape, 0, dtype=np.int32)
        left_x = int(max(original_center_x - new_width / 2, 0))
        right_x = int(min(original_center_x + new_width / 2, original_width))
        top_y = int(max(original_center_y - new_height / 2, 0))
        bottom_y = int(min(original_center_y + new_height / 2, original_height))

        canvas[top_y:bottom_y, left_x:right_x, :] = new_img[
            new_top_y:new_bottom_y, new_left_x:new_right_x, :
        ]
        return canvas

    def rotate_bbox(self, origin, bboxes):
        origin_x, origin_y = origin
        origin_y *= -1
        new_bbox = []

        for bbox in bboxes:
            tmp = []
            for points in bbox:
                x, y = self.rotate_point(
                    (origin_x, origin_y), (points[0], -1 * points[1])
                )
                tmp.append([x, -y])
            new_bbox.append(tmp)
        return new_bbox

    def __call__(self, image, bbox=None, label=None):
        self.angle = random.randint(1, 359)
        h, w, _ = image.shape
        rotate = self.rotate_image(image)
        new_img = self.crop_to_center(image, rotate)
        bboxes = self.rotate_bbox((w / 2, h / 2), bbox)
        labels = label
        # return self.rotate_bbox((w/2,h/2),bbox),self.rotate_image(image)

        for (bbox, anno) in zip(bboxes[:], labels[:]):
            out = False
            for pts in bbox:
                if pts[0] >= w or pts[0] < 0 or pts[1] >= h or pts[1] < 0:
                    out = True
                    break
            if out:
                bbox_copy = np.array(bbox, dtype=np.int32)
                # new_img = cv2.polylines(new_img, [bbox], True, (255, 255, 255), 0)
                new_img = cv2.fillConvexPoly(new_img, bbox_copy, [255, 255, 255])
                bboxes.remove(bbox)
                labels.remove(anno)

        if len(labels) == 0:
            return image, bbox, label
        return new_img, bboxes, labels


class geometry:
    def __init__(self):
        pass

    def hflip(self, image, bbox):
        transform = A.HorizontalFlip(p=1)
        length = len(bbox)
        width = image.shape[1]
        new_box = np.zeros((length, 4, 2))
        for i in range(length):
            for j in range(4):
                new_box[i][j][0] = width - bbox[i][j][0]
                new_box[i][j][1] = bbox[i][j][1]
        return transform(image=image)["image"], new_box

    def vflip(self, image, bbox):
        transform = A.VerticalFlip(p=1)
        length = len(bbox)
        height = image.shape[0]
        new_box = np.zeros((length, 4, 2))
        for i in range(length):
            for j in range(4):
                new_box[i][j][1] = height - bbox[i][j][1]
                new_box[i][j][0] = bbox[i][j][0]
        return transform(image=image)["image"], new_box

    def __call__(self, image, bbox):
        transform = random.sample([self.hflip, self.vflip], 1)[0]
        return transform(image, bbox)


class noise:
    def __init__(self):
        self.transform = A.OneOf(
            [
                A.GaussNoise(p=1),
            ],
            p=1,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class blur:
    def __init__(self):
        self.transform = A.OneOf(
            [
                # A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
                # A.Defocus(p=1),
                # A.GlassBlur(p=1, max_delta=1, iterations=1),
            ],
            p=1,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class weather:
    def __init__(self):
        self.transform = A.OneOf(
            [
                # A.RandomRain(p=1),
                # A.RandomSnow(p=1),
                # A.RandomFog(p=1),
                A.RandomShadow(p=1),
            ],
            p=1,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class camera:
    def __init__(self):
        self.transform = A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.JpegCompression(p=1),
            ],
            p=1,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class process:
    def __init__(self):
        self.transform = A.OneOf(
            [
                A.Posterize(p=1),
                A.Equalize(p=1),
                A.Solarize(p=1),
                A.InvertImg(p=1),
            ],
            p=1,
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


class augment:
    def __init__(self, img_size, crop_size):
        self.img_size = img_size
        self.crop_size = crop_size
        self.geometry = geometry()
        self.blur = blur()
        self.noise = noise()
        self.weather = weather()
        self.camera = camera()
        self.process = process()
        self.rotation = Rotate()

    def _resize(self, img: np.array, annotation):
        h, w, _ = img.shape
        size = self.img_size
        ratio = size / max(h, w)
        if w > h:
            img = A.Resize(int(h * ratio), size)(image=img)["image"]
        else:
            img = A.Resize(size, (int(w * ratio)))(image=img)["image"]
        for ann in annotation:
            for pts in ann:
                pts[0] *= ratio
                pts[1] *= ratio
        return img, annotation

    def _crop(self, img: np.array, annotation, labels):
        length = self.crop_size
        h, w, _ = img.shape
        # confirm the shortest side of image >= length
        if h >= w and w < length:
            img = A.Resize(int(h * length / w), length)
        elif h < w and h < length:
            img = img.resize(length, int(w * length / h))
        ratio_w = img.shape[1] / w
        ratio_h = img.shape[0] / h
        assert ratio_w >= 1 and ratio_h >= 1

        for ann in annotation:
            for pts in ann:
                pts[0] *= float(ratio_w)
                pts[1] *= float(ratio_h)
        remain_h = img.shape[0] - length
        remain_w = img.shape[1] - length

        cnt = 0
        flag = True

        while flag and cnt < 1000:
            cnt += 1
            start_w = int(np.random.rand() * remain_w)
            start_h = int(np.random.rand() * remain_h)
            for ann in annotation:
                check = True
                for pts in ann:
                    if (
                        pts[0] >= start_w
                        and pts[0] < start_w + length
                        and pts[1] >= start_h
                        and pts[1] < start_h + length
                    ):
                        continue
                    else:
                        check = False
                        break
                if check:
                    flag = False
                    break
            if not flag:
                for (ann, label) in zip(annotation[:], labels[:]):
                    out = False
                    for pts in ann:
                        if not (
                            pts[0] >= start_w
                            and pts[0] < start_w + length
                            and pts[1] >= start_h
                            and pts[1] < start_h + length
                        ):
                            out = True
                            break
                    if out:
                        tmp = np.array(ann, dtype=np.int32)
                        img = cv2.fillConvexPoly(img, tmp, [255, 255, 255])
                        annotation.remove(ann)
                        labels.remove(label)

                for ann in annotation:
                    for pts in ann:
                        pts[0] -= float(start_w)
                        pts[1] -= float(start_h)

        if flag == False:
            img = img[start_h : start_h + length, start_w : start_w + length, :]
        return img, annotation, labels

    def __call__(self, img, annotation, label):
        transform_list = random.sample(
            [
                self.geometry,
                self.noise,
                self.weather,
                self.camera,
                self.process,
                self.blur,
            ],
            2,
        )
        for transform in transform_list:
            if transform == self.geometry:
                img, annotation = transform(img, annotation)
            else:
                img = transform(img)

        img, annotation = self._resize(img, annotation)
        img, annotation, label = self.rotation(img, annotation, label)
        img, annotation, label = self._crop(img, annotation, label)
        # geometry적용시 bbox, annotation 변경
        return dict(img=img, annotation=annotation, label=label)


# Rotate

# Perpective

# Distort

#
