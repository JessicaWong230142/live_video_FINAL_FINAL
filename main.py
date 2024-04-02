# there is cited code in the lane_detection.py file!
import cv2
import numpy as np
import os
from GUI import window

file_path = "USER.db"

if not os.path.exists(file_path):
    open(file_path, 'w').close()

window() #call window function
