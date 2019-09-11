import os
import requests
import random
import _thread as thread
from uuid import uuid4
import urllib.parse as urlparse

import numpy as np
import skimage
from skimage.filters import gaussian

import zipfile
from pyunpack import Archive
from PIL import Image
import matplotlib.image as mpimg
import cv2


def blur(image, x0, x1, y0, y1, sigma=1, multichannel=True):
    y0, y1 = min(y0, y1), max(y0, y1)
    x0, x1 = min(x0, x1), max(x0, x1)
    im = image.copy()
    sub_im = im[y0:y1,x0:x1].copy()
    blur_sub_im = gaussian(sub_im, sigma=sigma, multichannel=multichannel)
    blur_sub_im = np.round(255 * blur_sub_im)
    im[y0:y1,x0:x1] = blur_sub_im
    return im



def download(url, filename):
    data = requests.get(url).content
    with open(filename, 'wb') as handler:
        handler.write(data)

    return filename


def generate_random_filename(upload_directory, extension):
    filename = str(uuid4())
    filename = os.path.join(upload_directory, filename + "." + extension)
    return filename


def clean_me(filename):
    if os.path.exists(filename):
        os.remove(filename)


def clean_all(files):
    for me in files:
        clean_me(me)


def create_directory(path):
    os.system("mkdir -p %s" % os.path.dirname(path))


def get_model_bin(url, output_path):
    if not os.path.exists(output_path):
        create_directory(output_path)
        filename, ext = os.path.splitext(os.path.basename(urlparse.urlsplit(url).path))
        if not os.path.exists(os.path.join(output_path, filename, ext)):
            print("downloading model :" + filename + ext)
            cmd = "wget -O %s %s" % (output_path, url)
            os.system(cmd)

    return output_path


#model_list = [(url, output_path), (url, output_path)]
def get_multi_model_bin(model_list):
    for m in model_list:
        thread.start_new_thread(get_model_bin, m)


def unzip(path_to_zip_file, directory_to_extract_to='.'):
    print("deflating model :" + path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def unrar(path_to_rar_file, directory_to_extract_to='.'):
    print("deflating model :" + path_to_rar_file)
    Archive(path_to_rar_file).extractall(directory_to_extract_to)


def resize_img_in_folder(path, w, h):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((w, h), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)


def resize_img(path, w, h):
    img = mpimg.imread(path)
    img = cv2.resize(img, dsize=(w, h))
    return img

def square_center_crop(image_path, output_path):
    im = Image.open(image_path)

    width, height = im.size

    new_width = min(width, height)
    new_height = new_width

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2



def image_crop(image_path, output_path, x0, y0, x1, y1):
    """
    The syntax is the following:
    cropped = img.crop( ( x, y, x + width , y + height ) )

    x and y are the top left coordinate on image;
    x + width and y + height are the width and height respectively of the region that you want to crop starting at x and ypoint.
    Note: x + width and y + height are the bottom right coordinate of the cropped region.
    """
    
    image = cv2.imread(image_path)

    print(x0, y0, x1, y1)
    crop = image[y0:y1, x0:x1]

    print(crop)

    cv2.imwrite(output_path, crop)

