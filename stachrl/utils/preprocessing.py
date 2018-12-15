import cv2


def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def crop(image, up, down, left, right):
    return image[up:(-1 - down), left:(-1 - right)]
