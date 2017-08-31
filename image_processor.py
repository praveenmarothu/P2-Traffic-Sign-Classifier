from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform,AffineTransform
import random
import numpy as np
import cv2


class ImageProcessor(object):

    @classmethod
    def random_transform(cls,img):
        image_size = img.shape[0]
        d = image_size * 0.2
        tl_top,tl_left,bl_bottom,bl_left,tr_top,tr_right,br_bottom,br_right = np.random.uniform(-d, d,size=8)   # Bottom right corner, right margin
        aft=  AffineTransform(scale=(1, 1/1.2))
        img= warp(img, aft,output_shape=(image_size, image_size), order = 1, mode = 'edge')
        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))

        img = warp(img, transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')
        return img

    @classmethod
    def grayscale_normalize(cls,img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = clahe.apply(img)
        img= (np.float32(img)-np.min(img))/( np.max(img) - np.min(img) )
        return img