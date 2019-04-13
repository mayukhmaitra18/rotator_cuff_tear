import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

pathIn = './images/'
files = [f for f in os.listdir(pathIn)]
files.sort()
path_1 = '/Users/mayukhmaitra/PycharmProjects/OpenCV/sift_res'
for images in os.listdir(pathIn):

    inp_image = cv2.imread('./images/' + str(images), 0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(inp_image, None)

    img_kp = cv2.drawKeypoints(inp_image, kp, inp_image)

    for images_1 in os.listdir(pathIn):
        inp_image_1 = cv2.imread('./images/' + str(images_1), 0)
        if(images != images_1):
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(inp_image_1, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des, des1, k=2)

            matchesMask = [[0, 0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.55 * n.distance:
                    matchesMask[i] = [1, 0]

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)

            img3 = cv2.drawMatchesKnn(inp_image, kp, inp_image_1, kp1, matches, None, **draw_params)
            result = 'res_' + str(images) + '_' + str(images_1)
            cv2.imwrite(os.path.join(path_1, result), img3)
            #cv2.imshow("Resultant Image", img_matches)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

