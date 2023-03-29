import cv2
from skimage.color import rgb2hsv,hsv2rgb
import numpy as np

from global_Stretching import global_stretching2 as global_stretching


def  HSVStretching(sceneRadiance):
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = global_stretching(s, height, width)
    img_v_stretching = global_stretching(v, height, width)

    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255

    # img_rgb = np.clip(img_rgb, 0, 255)

    return img_rgb


def  HSVStretching2(sceneRadiance):
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    img_hsv[:, :, 1] = global_stretching(img_hsv[:, :, 1], height, width)
    img_hsv[:, :, 2] = global_stretching(img_hsv[:, :, 2], height, width)
    img_rgb = hsv2rgb(img_hsv) * 255

    return img_rgb