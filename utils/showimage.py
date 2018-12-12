import matplotlib.pyplot as plt


def showimage(image, gray=True):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
