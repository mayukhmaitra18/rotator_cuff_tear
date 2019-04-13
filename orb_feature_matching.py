import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

pathIn = './images/'
files = [f for f in os.listdir(pathIn)]
files.sort()
path_1 = '/Users/mayukhmaitra/PycharmProjects/OpenCV/orb_res'
for images in os.listdir(pathIn):
    inp_image = cv2.imread('./images/' + str(images), 0)
    orb = cv2.ORB_create()  # OpenCV 3 backward incompatibility: Do not create a detector with `cv2.ORB()`.
    key_points, description = orb.detectAndCompute(inp_image, None)

    for images_1 in os.listdir(pathIn):
        inp_image_1 = cv2.imread('./images/' + str(images_1), 0)
        if(images != images_1):
            orb = cv2.ORB_create()  # OpenCV 3 backward incompatibility: Do not create a detector with `cv2.ORB()`.
            key_points_1, description_1 = orb.detectAndCompute(inp_image, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(description, description_1)
            matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

            img_matches = cv2.drawMatches(inp_image, key_points, inp_image_1, key_points_1, matches[:100], inp_image_1,
                                          flags=2)  # Show top 10 matches
            result = 'res_' + str(images) + '_' + str(images_1)
            cv2.imwrite(os.path.join(path_1, result), img_matches)
            #cv2.imshow("Resultant Image", img_matches)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

