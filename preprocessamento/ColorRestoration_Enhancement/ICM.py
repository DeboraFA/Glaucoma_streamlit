import os
import numpy as np
import cv2
import natsort
import xlwt
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB0 as sceneRadianceRGB


np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def ICM(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = stretching(img)
    sceneRadiance = sceneRadianceRGB(img)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    return sceneRadiance

from PIL import Image
def ICM2(uploaded_file):
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = stretching(img)
    sceneRadiance = sceneRadianceRGB(img)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    return sceneRadiance
