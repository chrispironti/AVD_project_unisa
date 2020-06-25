"""
#
# Adapted from https://github.com/DamonMIN/Traffic-Lights-Detection-and-Recognition
#
"""

import cv2
import numpy as np
from skimage import segmentation,measure



strLight = ""
FONT = cv2.FONT_HERSHEY_SIMPLEX

def classify(mask, opened, cimg, color):
    """
    Determines if an image may contain a traffic light,
    using a specific color mask and its opening image
    :param mask: color mask
    :param opened: color mask after opening operation
    :param cimg: original image
    :param color: corresponding mask's color
    :return: labelled image if traffic light is found, else None
    """


    # Label every connected regions
    label_image  = measure.label(opened)

    # Finding edges
    borders  = np.logical_xor(mask, opened)

    # Writing edges
    label_image [borders ] = -1

    # Iterate over every labelled region
    for region  in measure.regionprops(label_image ):

        # Basic convexity condition
        if region .convex_area < 120 or region .area > 2000:
            continue

        # Compute basic metrics
        area = region .area                     # Number of pixels of the region
        eccentricity = region .eccentricity     # Eccentricity of the ellipse that has the same second-moments as the region
        convex_area  = region .convex_area      # Number of pixels of the smallest convex polygon that encloses the region
        minr, minc, maxr, maxc = region .bbox   # Bounding boxes coordinates
        radius = int(max(maxr-minr,maxc-minc)/2)# Radius of the bounding box
        centroid = region .centroid             # Centroid coordinate tuple
        perimeter    = region .perimeter        # Perimeter

        # Centroid coordinates
        x = int(centroid[0])
        y = int(centroid[1])

        # if region is a point
        if perimeter == 0:
            circularity = 1
            circum_circularity = 0
        else:
            # Compute advanced metrics
            circularity = 12.56*area/(perimeter*perimeter)
            circum_circularity      = 12.56*convex_area/(39.92*radius*radius)

        # Check if the current regions verifies one of the 3 conditions
        if eccentricity <= 0.4 or circularity >= 0.7 or circum_circularity >= 0.73:
            cv2.circle(cimg, (y,x), radius, (0,0,255),3)
            cv2.putText(cimg,color,(y,x), FONT, 1,(0,0,255),2)
            # Some traffic light detected
            return cimg
        else:
            # No traffic light detected
            return None


def detect(img):
    """

    :param image: Cropped image of a traffic light with margin
    :return: traffic light color as a string, image with label
    """


    # Definition of the color ranges in HSV color space
    # Two Red HSV color ranges that will be combined
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])

    # Green HSV color range
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])

    # Yellow HSV color range
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])


    # Convert RGB image to HSV
    cimg = np.float32(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Apply masks
    mask1 = cv2.inRange(hsv, lower_red1,   upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2,   upper_red2)
    mask_g = cv2.inRange(hsv, lower_green,  upper_green)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_r = cv2.add(mask1, mask2) # Combine the two red masks

    # Defining the Structuring element that will be used in the Opening operation
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))


    # Process to opening morphological operation
    opened_r  = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, element)
    opened_g  = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, element)
    opened_y  = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, element)

    # Verify if the red mask may contain an element corresponding to a traffic light
    is_red = classify(mask_r, opened_r, cimg, 'RED')
    if is_red is not None:
        return 'RED', is_red

    # Same with Green
    is_green = classify(mask_g, opened_g, cimg, 'GREEN')
    if is_green is not None:
        return 'GREEN', is_green

    # Same with Yellow
    is_yellow = classify(mask_y, opened_y, cimg, 'YELLOW')
    if is_yellow is not None:
        return 'YELLOW', is_yellow

    # None if any of the masks may correspond to a traffic light
    return "NONE",img



def max(a, b):
    # returns max of a and b
    if a>b:
        return a
    else: 
        return b



def detect_color(image):
    """
    Main function to be called passing a cropped traffic light image with margin
    You may change the signature to return the image array with
    the traffic light labeled
    :param image: Cropped image of a traffic light with margin
    :return: traffic light color as a string
    """
    strLight,result = detect(image)
    # return detect(image)
    return strLight







# ==============================================================================
# -- Testing --------------------------------------------------------
# ==============================================================================

#
# def test_dir(dir):
#     for filename in os.listdir(dir):
#         if filename.endswith(".png") :
#             #beg = datetime.datetime.now()
#             img = cv2.imread(dir+filename)
#             #end = datetime.datetime.now()
#
#             print(detect_color(img))
#             continue
#         else:
#             continue
#
# dir= "F:\\Data\\cours\\AVD\\TLdataset01\\test\\" #specify directory
#
# test_dir(dir)