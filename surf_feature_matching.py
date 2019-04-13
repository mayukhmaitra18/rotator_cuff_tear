import numpy
import cv2
import os



#surf and match
def image_matching(input_image_1, input_image_2):
    """Given two images, returns the basic_match"""
    surf = cv2.xfeatures2d.SURF_create(400, 5, 5)
    bf_match = cv2.BFMatcher(cv2.NORM_L2)

    keypoint_1, desc1 = surf.detectAndCompute(input_image_1, None)
    keypoint_2, desc2 = surf.detectAndCompute(input_image_2, None)

    basic_match = bf_match.knnMatch(desc1, trainDescriptors = desc2, k = 2)
    keypoint_pairs = filters(keypoint_1, keypoint_2, basic_match)
    return keypoint_pairs

def filters(keypoint_1, keypoint_2, basic_match, ratio = 0.75):
    kp1_matrix, kp2_matrix = [], []
    for m in basic_match:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            kp1_matrix.append( keypoint_1[m.queryIdx] )
            kp2_matrix.append( keypoint_2[m.trainIdx] )
    keypoint_pairs = zip(kp1_matrix, kp2_matrix)
    return keypoint_pairs


#alignment and display
def explore_match(win, input_image_1, input_image_2, keypoint_pairs,images, images_1, status = None, H = None ):
    height_1, width_1 = input_image_1.shape[:2]
    height_2, width_2 = input_image_2.shape[:2]
    output_img = numpy.zeros((max(height_1, height_2), width_1+width_2), numpy.uint8)
    output_img[:height_1, :width_1] = input_image_1
    output_img[:height_2, width_1:width_1+width_2] = input_image_2
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = numpy.float32([[0, 0], [width_1, 0], [width_1, height_1], [0, height_1]])
        corners = numpy.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (width_1, 0) )
        cv2.polylines(output_img, [corners], True, (255, 255, 255))

    if status is None:
        status = numpy.ones(len(keypoint_pairs), numpy.bool_)
    p1 = numpy.int32([kpp[0].pt for kpp in keypoint_pairs])
    p2 = numpy.int32([kpp[1].pt for kpp in keypoint_pairs]) + (width_1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (i, j), (k, l), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(output_img, (i, j), 2, col, -1)
            cv2.circle(output_img, (k, l), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(output_img, (i-r, j-r), (i+r, j+r), col, thickness)
            cv2.line(output_img, (i-r, j+r), (i+r, j-r), col, thickness)
            cv2.line(output_img, (k-r, l-r), (k+r, l+r), col, thickness)
            cv2.line(output_img, (k-r, l+r), (k+r, l-r), col, thickness)
    output_img0 = output_img.copy()
    for (i, j), (k, l), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(output_img, (i, j), (k, l), green)

    result = 'res_' + str(images) + '_' + str(images_1)
    cv2.imwrite(os.path.join('/Users/mayukhmaitra/PycharmProjects/OpenCV', result), output_img)
    cv2.imshow(win, output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#output driver function
def output_match(window_name, keypoint_pairs, input_image_1, input_image_2, images, images_1):
    """Draws the basic_match for """
    kp1_matrix, kp2_matrix = zip(*keypoint_pairs)

    p1 = numpy.float32([kp.pt for kp in kp1_matrix])
    p2 = numpy.float32([kp.pt for kp in kp2_matrix])

    if len(keypoint_pairs) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    else:
        H, status = None, None
    if len(p1):
        explore_match(window_name, input_image_1, input_image_2, keypoint_pairs,images, images_1, status, H )


#driver function

pathIn = './images/'

for images in os.listdir(pathIn):
    input_image_1 = cv2.imread('./images/' + str(images), 0)
    for images_1 in os.listdir(pathIn):
        input_image_2 = cv2.imread('./images/' + str(images_1), 0)
        if(images != images_1):
            #input_image_1 = cv2.imread('./images/a.png', 0)
            #input_image_2 = cv2.imread('./images/b.png', 0)
            keypoint_pairs = image_matching(input_image_1, input_image_2)
            if keypoint_pairs:
                output_match('features matched', keypoint_pairs, input_image_1, input_image_2, images, images_1)
            else:
                print("No match between these two images")
