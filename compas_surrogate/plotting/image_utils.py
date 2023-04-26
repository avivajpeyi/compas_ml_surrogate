import os
from typing import List

import numpy as np
from PIL import Image


def change_height_proportionally(img, width):
    """Change height of image proportional to given width."""
    wpercent = width / img.size[0]
    proportional_height = int(img.size[1] * wpercent)
    return img.resize((width, proportional_height), Image.ANTIALIAS)


def change_width_proportionally(img, height):
    """Change width of image proportional to given height."""
    hpercent = height / img.size[1]
    proportional_width = int(img.size[0] * hpercent)
    return img.resize((proportional_width, height), Image.ANTIALIAS)


def make_same_width(image_list):
    """Make all images in input list the same width."""
    imgs = [Image.open(i) for i in image_list]
    min_width = min([i.size[0] for i in imgs])
    resized = [change_height_proportionally(img, min_width) for img in imgs]
    return [np.asarray(i) for i in resized]


def make_same_height(image_list):
    """Make all images in input list the same height."""
    imgs = [Image.open(i) for i in image_list]
    min_height = min([i.size[1] for i in imgs])
    resized = [change_width_proportionally(img, min_height) for img in imgs]
    return [np.asarray(i) for i in resized]


def horizontal_concat(png_list: List[str], savefn: str, rm_orig=False):
    im = Image.fromarray(np.hstack(make_same_height(png_list)))
    im.save(savefn, dpi=im.size)
    if rm_orig:
        remove_files(png_list)


def vertical_concat(png_list: List[str], savefn: str, rm_orig=False):
    im = Image.fromarray(np.vstack(make_same_width(png_list)))
    im.save(savefn, dpi=im.size)
    if rm_orig:
        remove_files(png_list)


def remove_files(file_list):
    for f in file_list:
        os.remove(f)
