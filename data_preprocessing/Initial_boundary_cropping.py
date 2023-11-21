from matplotlib import pyplot as plt


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


