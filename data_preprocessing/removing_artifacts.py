import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt


def ShowHist255(img, ignore_zero=False):
    hist, bin_edges = np.histogram(img, bins=255, density=False)

    if ignore_zero:
        plt.plot(bin_edges[1:-1], hist[1:])
    else:
        plt.plot(bin_edges[0:-1], hist)

    plt.show()


def MinMaxNormalise(img):
    '''
    This function does min-max normalisation on
    the given image.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to normalise.

    Returns
    -------
    norm_img: {numpy.ndarray}
        The min-max normalised image.
    '''

    norm_img = (img - img.min()) / (img.max() - img.min())

    return norm_img


def CropBorders(img):
    '''
    This function crops 1% from all four sides of the given
    image.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to crop.

    Returns
    -------
    cropped_img: {numpy.ndarray}
        The cropped image.
    '''
    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * 0.01)
    r_crop = int(ncols * (1 - 0.04))
    u_crop = int(nrows * 0.01)
    d_crop = int(nrows * (1 - 0.04))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]

    return cropped_img



def Binarisation(img, maxval, show=False):
    # First convert image to uint8.
    img = skimage.img_as_ubyte(img)

    thresh, th1 = cv2.threshold(src=img,
                                thresh=0.1,
                                maxval=maxval,
                                type=cv2.THRESH_BINARY)  # Global thresholding

    otsu_thresh, th2 = cv2.threshold(src=img,
                                     thresh=0,
                                     maxval=maxval,
                                     type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's thresholding

    th3 = cv2.adaptiveThreshold(src=img,
                                maxValue=maxval,
                                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                thresholdType=cv2.THRESH_BINARY,
                                blockSize=9,
                                C=-1)

    th4 = cv2.adaptiveThreshold(src=img,
                                maxValue=maxval,
                                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                thresholdType=cv2.THRESH_BINARY,
                                blockSize=9,
                                C=-1)

    images = [img, th1, th2, th3, th4]
    titles = ['Original Image',
              'Global Thresholding (v = 0.1)',
              'Global Thresholding (otsu)',
              'Adaptive Mean Thresholding',
              'Adaptive Gaussian Thresholding']

    # --- Plot the different thresholds ---
    if show:
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(22, 5))

        for i in range(5):
            ax[i].imshow(images[i], cmap="gray")
            ax[i].set_title(titles[i])
        plt.show()

    return th1, th2, th3, th4


def DilateMask(mask):
    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return dilated_mask


def ApplyMask(img, dilated_mask):
    # Apply change
    result = img.copy()
    result[dilated_mask == 0] = 0  # We only keep areas that are white (255)

    return result


def OwnGlobalBinarise(img, thresh, maxval):
    '''
    This function takes in a numpy array image and
    returns a corresponding mask that is a global
    binarisation on it based on a given threshold
    and maxval. Any elements in the array that is
    greater than or equals to the given threshold
    will be assigned maxval, else zero.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to perform binarisation on.
    thresh : {int or float}
        The global threshold for binarisation.
    maxval : {np.uint8}
        The value assigned to an element that is greater
        than or equals to `thresh`.


    Returns
    -------
    binarised_img : {numpy.ndarray, dtype=np.uint8}
        A binarised image of {0, 1}.
    '''

    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval

    return binarised_img



def OpenMask(mask, ksize=(23, 23), operation="open"):
    '''
    This function edits a given mask (binary image) by performing
    closing then opening morphological operations.

    Parameters
    ----------
    mask : {numpy.ndarray}
        The mask to edit.

    Returns
    -------
    edited_mask : {numpy.ndarray}
        The mask after performing close and open morphological
        operations.
    '''

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    return edited_mask


def SortContoursByArea(contours, reverse=True):
    '''
    This function takes in list of contours, sorts them based
    on contour area, computes the bounding rectangle for each
    contour, and outputs the sorted contours and their
    corresponding bounding rectangles.

    Parameters
    ----------
    contours : {list}
        The list of contours to sort.

    Returns
    -------
    sorted_contours : {list}
        The list of contours sorted by contour area in descending
        order.
    bounding_boxes : {list}
        The list of bounding boxes ordered corresponding to the
        contours in `sorted_contours`.
    '''

    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes


def DrawContourID(img, bounding_box, contour_id):
    '''
    This function draws the given contour and its ID on the given
    image. The image with the drawn contour is returned.

    Parameters
    ----------
    img: {numpy.ndarray}
        The image to draw the contour on.
    bounding_box : {tuple of int}
        The bounding_rect of the given contour.
    contour_id : {int or float}
        The corresponding ID of the given `contour`.

    Returns
    -------
    img : {numpy.ndarray}
        The image after the `contour` and its ID is drawn on.
    '''

    # Center of bounding_rect.
    x, y, w, h = bounding_box
    center = (((x + w) // 2), ((y + h) // 2))

    # Draw the countour number on the image
    cv2.putText(img=img,
                text=f"{contour_id}",
                org=center,  # Bottom-left corner of the text string in the image.
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=10,
                color=(255, 255, 255),
                thickness=40)

    return img


def XLargestBlobs(mask, top_X=None):
    '''
    This function finds contours in the given image and
    keeps only the top X largest ones.

    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to get the top X largest blobs.
    top_X : {int}
        The top X contours to keep based on contour area
        ranked in decesnding order.


    Returns
    -------
    n_contours : {int}
        The number of contours found in the given `mask`.
    X_largest_blobs : {numpy.ndarray}
        The corresponding mask of the image containing only
        the top X largest contours in white.
    '''

    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)

    n_contours = len(contours)

    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:

        # Make sure that the number of contours to keep is at most equal
        # to the number of contours present in the mask.
        if n_contours < top_X or top_X == None:
            top_X = n_contours

        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = SortContoursByArea(contours=contours,
                                                             reverse=True)

        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_X]

        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)

        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(image=to_draw_on,  # Draw the contours on `to_draw_on`.
                                           contours=X_largest_contours,  # List of contours to draw.
                                           contourIdx=-1,  # Draw all contours in `contours`.
                                           color=1,  # Draw the contours in white.
                                           thickness=-1)  # Thickness of the contour lines.

    return n_contours, X_largest_blobs


def InvertMask(mask):
    '''
    This function inverts a given mask (i.e. 0 -> 1
    and 1 -> 0).

    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to invert.

    Returns
    -------
    inverted_mask: {numpy.ndarray}
        The inverted mask.
    '''

    inverted_mask = np.zeros(mask.shape, np.uint8)
    inverted_mask[mask == 0] = 1

    return inverted_mask


def InPaint(img, mask, flags="telea", inpaintRadius=30):
    '''
    This function restores an input image in areas indicated
    by the given mask (elements with 1 are restored).

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to restore.
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask that indicates where (elements == 1) in the
        `img` the damage is.
    inpaintRadius : {int}
        Radius of a circular neighborhood of each point
        inpainted that is considered by the algorithm.

    Returns
    -------
    inpainted_img: {numpy.ndarray}
        The restored image.
    '''

    # First convert to `img` from float64 to uint8.
    img = 255 * img
    img = img.astype(np.uint8)

    # Then inpaint based on flags.
    if flags == "telea":
        inpainted_img = cv2.inpaint(src=img,
                                    inpaintMask=mask,
                                    inpaintRadius=inpaintRadius,
                                    flags=cv2.INPAINT_TELEA)
    elif flags == "ns":
        inpainted_img = cv2.inpaint(src=img,
                                    inpaintMask=mask,
                                    inpaintRadius=inpaintRadius,
                                    flags=cv2.INPAINT_NS)

    return inpainted_img


def ApplyMask(img, mask):
    '''
    This function applies a mask to a given image. White
    areas of the mask are kept, while black areas are
    removed.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to mask.
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to apply.

    Returns
    -------
    masked_img: {numpy.ndarray}
        The masked image.
    '''

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    return masked_img


def HorizontalFlip(mask):
    '''
    This function figures out how to flip (also entails whether
    or not to flip) a given mammogram and its mask. The correct
    orientation is the breast being on the left (i.e. facing
    right) and it being the right side up. i.e. When the
    mammogram is oriented correctly, the breast is expected to
    be found in the bottom left quadrant of the frame.

    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The corresponding mask of the CC image to flip.

    Returns
    -------
    horizontal_flip : {boolean}
        True means need to flip horizontally,
        False means otherwise.
    '''

    # Get number of rows and columns in the image.
    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2

    # Sum down each column.
    col_sum = mask.sum(axis=0)
    # Sum across each row.
    row_sum = mask.sum(axis=1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])
    top_sum = sum(row_sum[0:y_center])
    bottom_sum = sum(row_sum[y_center:-1])

    if left_sum < right_sum:
        horizontal_flip = True
    else:
        horizontal_flip = False

    return horizontal_flip


def clahe(img, clip=2.0, tile=(8, 8)):
    '''
    This function applies the Contrast-Limited Adaptive
    Histogram Equalisation filter to a given image.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to edit.
    clip : {int or floa}
        Threshold for contrast limiting.
    tile : {tuple (int, int)}
        Size of grid for histogram equalization. Input
        image will be divided into equally sized
        rectangular tiles. `tile` defines the number of
        tiles in row and column.

    Returns
    -------
    clahe_img : {numpy.ndarray}
        The edited image.
    '''

    # Convert to uint8.
    # img = skimage.img_as_ubyte(img)
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img


def Pad(img):
    '''
    This function pads a given image with black pixels,
    along its shorter side, into a square and returns
    the square image.

    If the image is portrait, black pixels will be
    padded on the right to form a square.

    If the image is landscape, black pixels will be
    padded on the bottom to form a square.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to pad.

    Returns
    -------
    padded_img : {numpy.ndarray}
        The padded square image, if padding was required
        and done.
    img : {numpy.ndarray}
        The original image, if no padding was required.
    '''

    nrows, ncols = img.shape

    # If padding is required...
    if nrows != ncols:

        # Take the longer side as the target shape.
        if ncols < nrows:
            target_shape = (nrows, nrows)
        elif nrows < ncols:
            target_shape = (ncols, ncols)

        # Pad.
        padded_img = np.zeros(shape=target_shape)
        padded_img[:nrows, :ncols] = img

        return padded_img

    # If padding is not required, return original image.
    elif nrows == ncols:

        return img
