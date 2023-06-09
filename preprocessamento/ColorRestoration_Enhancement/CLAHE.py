import os
import numpy as np
import cv2
import natsort
import xlwt
from skimage import exposure

from sceneRadianceCLAHE import RecoverCLAHE
from sceneRadianceHE import RecoverHE

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def CLAHE(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sceneRadiance = RecoverCLAHE(img)
    return sceneRadiance

from PIL import Image

def CLAHE2(uploaded_file):
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sceneRadiance = RecoverCLAHE(img)
    return sceneRadiance
