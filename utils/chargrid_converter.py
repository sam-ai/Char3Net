import pytesseract
import cv2
from pytesseract import Output
import numpy as np
import os


CHAR_MAP = {
    '"': (0, 255, 0),
    '#': (0, 255, 0),
    '%': (0, 255, 0),
    '&': (0, 255, 0),
    ',': (0, 255, 0),
    '.': (0, 255, 0),
    '0': (0, 0, 255),
    '1': (0, 0, 255),
    '2': (0, 0, 255),
    '3': (0, 0, 255),
    '4': (0, 0, 255),
    '5': (0, 0, 255),
    '6': (0, 0, 255),
    '7': (0, 0, 255),
    '8': (0, 0, 255),
    '9': (0, 0, 255),
    ':': (0, 255, 0),
    'a': (0, 255, 255),
    'b': (0, 255, 255),
    'c': (0, 255, 255),
    'd': (0, 255, 255),
    'e': (0, 255, 255),
    'f': (0, 255, 255),
    'g': (0, 255, 255),
    'h': (0, 255, 255),
    'i': (0, 255, 255),
    'j': (0, 255, 255),
    'k': (0, 255, 255),
    'l': (0, 255, 255),
    'm': (0, 255, 255),
    'n': (0, 255, 255),
    'o': (0, 255, 255),
    'p': (0, 255, 255),
    'q': (0, 255, 255),
    'r': (0, 255, 255),
    's': (0, 255, 255),
    't': (0, 255, 255),
    'u': (0, 255, 255),
    'v': (0, 255, 255),
    'w': (0, 255, 255),
    'x': (0, 255, 255),
    'y': (0, 255, 255),
    'z': (0, 255, 255)
}


def char_grid(
        input_img_path,
        output_dir
):
    img = cv2.imread(input_img_path)
    height = img.shape[0]
    width = img.shape[1]
    blank_image = np.zeros(
        (height, width, 3),
        np.uint8
    )
    tess_img = pytesseract.image_to_boxes(
        img,
        output_type=Output.DICT
    )
    n_boxes = len(tess_img['char'])
    for i in range(n_boxes):
        try:
            annot_a = True
            (text, x1, y2, x2, y1) = (
                tess_img['char'][i],
                tess_img['left'][i],
                tess_img['top'][i],
                tess_img['right'][i],
                tess_img['bottom'][i]
            )
        except:
            annot_a = False

        if annot_a:
            char_color = CHAR_MAP.get(
                text.strip().lower(),
                (255, 255, 255)
            )
            x, y, w, h = x1, y1,\
                         (x2 - x1), (y2 - y1)

            cv2.rectangle(
                blank_image,
                (x1, height - y1),
                (x2, height - y2),
                char_color,
                cv2.FILLED
            )

            filename = input_img_path.split('/')[-1]
            cv2.imwrite(
                os.path.join(output_dir, filename),
                blank_image
            )







