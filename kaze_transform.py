import cv2
import numpy as np
import _pickle as pickle
import os


# Feature extractor
def feature_extraction(image_path, vector_size=32):
    image = cv2.imread(image_path,1)
    try:
        # Using KAZE transform to detect feature vectors
        alg = cv2.KAZE_create()
        
        # Detecting keypoints in image
        keypoints = alg.detect(image)
        
        # Number of keypoints varies depending on the size of image as well as color
        # Bigger keypoint response value is better, sorting according to that 
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]
        
        # computing descriptors vector
        keypoints, descriptor = alg.compute(image, keypoints)
        
        # Flattening the above vectors- the required feature vector
        descriptor = descriptor.flatten()
        
        
        # Descriptor vector size is 64
        desc_needed_size = (vector_size * 64)
        if descriptor.size < desc_needed_size:
            
            descriptor = np.concatenate([descriptor, np.zeros(desc_needed_size - descriptor.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return descriptor


def extractor_driver(images_path, pickled_file="kaze_features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:

        name = f.split('/')[-1].lower()
        result[name] = feature_extraction(f)

    # saving all our feature vectors in pickled file
    with open(pickled_file, 'wb') as fp:
        pickle.dump(result, fp)
    with open(pickled_file, 'rb') as fp:
        data = pickle.load(fp)
    image_names = []
    feature_matrix = []

    #storing the image names and their matrices
    for k, v in data.items():
        image_names.append(k)
        feature_matrix.append(v)
    image_names = np.array(image_names)
    feature_matrix = np.array(feature_matrix)

    print('features for images:',data)

def main():
    images_path = './images/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    extractor_driver(images_path)

main()