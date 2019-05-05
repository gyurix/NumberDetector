import os

import tensorflow as tf
from PIL import Image, ImageFilter


def load_img(filename):
    with Image.open(filename) as img:
        im = img.convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels
        if width > height:
            nheight = int(round((20.0 / width * height), 0))
            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))
            newImage.paste(img, (4, wtop))
        else:
            nwidth = int(round((20.0 / height * width), 0))
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))
            newImage.paste(img, (wleft, 4))
        data = list(newImage.getdata())
        ndata = [(255 - x) * 1.0 / 255.0 for x in data]
        return ndata


def has_valid_extension(fn):
    for ext in ['.jpg', '.png', '.bmp']:
        if fn.endswith(ext):
            return True
    return False


def process_images(folder):
    for file_name in os.listdir(folder):
        if not has_valid_extension(file_name):
            continue
        img = load_img(file_name)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
process_images('data')
