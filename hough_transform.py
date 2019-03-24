import cv2
import numpy as np
import os

path = './images'
for images in os.listdir(path):
    result = 'res_'+str(images)
    print(images)
    inputImage = cv2.imread('./images/'+str(images),1)
    inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    smoothImage = cv2.GaussianBlur(inputImageGray, (5, 5), 0)

    edges = cv2.Canny(smoothImage,150,200,apertureSize = 3)
    minLineLength = 1000
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            cv2.polylines(inputImage, [pts], True, (0,255,0))

    path_1 = './hough_res'
    cv2.imwrite(os.path.join(path_1, result), inputImage)
    cv2.imshow("Resultant Image", inputImage)
    #cv2.imshow('edge', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()