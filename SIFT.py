# import packages here
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os

pathIn = './images/'
files = [f for f in os.listdir(pathIn)]
files.sort()
path_1 = './SIFT_res'
for images in os.listdir(pathIn):

    result = 'res_' + str(images)

    inp_image = cv2.imread('./images/' + str(images), 0)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(inp_image, None)
    res1 = cv2.drawKeypoints(inp_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(path_1, result), res1)
    cv2.imshow("Resultant Image", res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for images_1 in os.listdir(pathIn):

        img_input_2 = cv2.imread('./images/'+str(images_1),0)

        sift_2 = cv2.xfeatures2d.SIFT_create()
        keypoints_2, descriptor_2 = sift_2.detectAndCompute(img_input_2, None)
        res2 = cv2.drawKeypoints(img_input_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        bf_match = cv2.BFMatcher()
        match = bf_match.knnMatch(descriptor, descriptor_2, k=2)

        # Apply ratio test
        good_matches = []  # Append filtered matches to this list
        good_matches_without_list = []
        for i, j in match:
            if i.distance < 0.1 * j.distance:
                good_matches.append([i])
                good_matches_without_list.append(i)

        # draw matching results with the given drawMatches function
        res3 = cv2.drawMatchesKnn(inp_image, keypoints, img_input_2, keypoints_2, good_matches, None, flags=2)
        #str_ouput = "SIFT feature match between" + str(images) + ' and ' + str(images_1)
        #cv2.imshow(str_ouput, res3)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

       #plt.subplot(1, 2, 2)
       #plt.imshow(res2, 'gray')
       #plt.title('rotated img')
       #plt.axis('off')
'''
       bf_match = cv2.BFMatcher()
       match = bf_match.knnMatch(descriptor, descriptor_2, k=2)

       # Apply ratio test
       good_matches = []  # Append filtered matches to this list
       good_matches_without_list = []
       for i, j in match:
           if i.distance < 0.1 * j.distance:
               good_matches.append([i])
               good_matches_without_list.append(i)

       # draw matching results with the given drawMatches function
       res3 = cv2.drawMatchesKnn(img_input, keypoints, img_input_2, keypoints_2, good_matches, None, flags=2)


       plt.figure(figsize=(12, 8))
       plt.imshow(res3)
       plt.title('matching')
       plt.axis('off')
'''
