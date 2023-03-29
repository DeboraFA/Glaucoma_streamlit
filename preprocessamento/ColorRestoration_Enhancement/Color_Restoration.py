import os
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image




from DCP import DCP
from GBdehazingRCorrection import GBdehazingRC
from IBLA import IBLA
from LowComplexityDCP import LowComplexityDCP
from MIP import MIP
from NewOpticalModel import NewOpticalModel
from UDCP import UDCP
from Rows import Rows
from ULAP import ULAP


def color_restoration(img_path, preprocess):

    if preprocess == 'DCP':
        sceneRadiance = DCP(img_path)
    elif preprocess == 'GBdehazingRCorrection':
        sceneRadiance = GBdehazingRC(img_path)
    elif preprocess == 'IBLA':
        sceneRadiance = IBLA(img_path) 
    elif preprocess == 'LowComplexityDCP':
        sceneRadiance = LowComplexityDCP(img_path) 
    elif preprocess == 'MIP':
        sceneRadiance = MIP(img_path) 
    elif preprocess == 'NewOpticalModel':
        sceneRadiance = NewOpticalModel(img_path) 
    elif preprocess == 'Rows':
        sceneRadiance = Rows(img_path)         
    elif preprocess == 'UDCP':
        sceneRadiance = UDCP(img_path) 
    elif preprocess == 'ULAP':
        sceneRadiance = ULAP(img_path) 
                
    return sceneRadiance
    
