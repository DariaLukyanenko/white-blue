import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

def extract_border_and_dilate(image_path, dilate_percent):
    # Step 1: Load the dermoscopy image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Steps 2: Extracting border and Dilating is omitted due to the manual nature
    
    # Step 3: Dilate the border by percentage of its area
    kernel_size = int(np.sqrt(img.shape[0]*img.shape[1])*dilate_percent)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(img, kernel, iterations=1)
    return img, dilated

def celebi_method(img, dilated):
    # Step 4: Extract region outside the dilated border
    mask = cv2.subtract(dilated, img)
    external = cv2.bitwise_and(img, img, mask=mask)

    # Steps 5-6: If R > 90 and R > B and R > G then mark the pixel as healthy skin
    r, g, b = cv2.split(img)
    healthy_skin_pixels = (r > 90) & (r > b) & (r > g)
    
    # Step 7: Set Rs as the mean of the red channel values for pixels marked healthy skin
    mean_red = np.mean(r[healthy_skin_pixels])

    # Steps 8-11: For each pixel in the image, classify pixel as BWS
    nB = b/(r + g + b)
    rR = r/mean_red
    bws_pixels = (nB >= 0.3) & (-194 <= rR) & (rR < -51)
    
    return bws_pixels

def madooei_method(image_path):
    # Step 1-4: Convert from sRGB to CIELAB
    img = cv2.imread(image_path, cv2.COLOR_BGR2Lab)
  
    # Step 5: Replace each pixel with superpixel representation
    superpixels = slic(img, n_segments=100, compactness=10)

    # Veil detection and Munsell color system needs additional data
    # And we cannot implement it without such data
    # We skip these steps here since it is not achievable without the original data and domain expertise

    # Step 14-16: For each segmented region find the best match from colour palette
    # And if the best match is within the threshold distance then classify as BWS

    # This is skipped since we cannot implement it without palette and threshold distance

    return img, superpixels

def combined_method(img_path):
    # Run the preprocessing method for Celebi
    img, dilated = extract_border_and_dilate(img_path, 0.1)
    
    # Run the Celebi method to get the BWS pixels
    bws_pixels = celebi_method(img, dilated)
    
    # Run the Madooei method and get the segmented image and superpixels
    img, superpixels = madooei_method(img_path)

    # Combine the BWS pixels and the superpixels in some way
    # This is application specific so it is left blank here
    #combined_result = combine_bws_and_superpixels(bws_pixels, superpixels)

    return bws_pixels, img, superpixels

bws_pixels, img, superpixels = combined_method('path_to_your_image.jpg')
