import cv2

def load_image(image_path) -> np.ndarray: 
    """
    Load an image and return a grayscale image
    """
    # Our image generation process wrote everything to disk as 3-channel, but our lightsources are white
    # and there are no colored surfaces on these models, so everything is grayscale. Collapse the 
    # three channels down to 1 here to reduce the computation required to train our models
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray 