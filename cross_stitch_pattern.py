""" Photos to Needlework Designs
"""
import cv2
import random
import scipy.stats
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from floss_colors import *

from optparse import OptionParser
import os 


def create_pattern(filename, num_colors, bg_color=0):

    BGR_img = cv2.imread(filename)
    RGB_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)

    gradient_map = compute_gradient_map(RGB_img)
    #display_image(gradient_map)

    mask = np.zeros(RGB_img.shape[:2], np.uint8)
    height = RGB_img.shape[0]
    width = RGB_img.shape[1]

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    search_rect = (0, 0, width - 1, height - 1)
    cv2.grabCut(gradient_map, mask, search_rect, bgdModel, fgdModel, 25, cv2.GC_INIT_WITH_RECT)
    
    bg_mask = np.where(mask == 2, 0, 1).astype('uint8')
    foreground = RGB_img * bg_mask[:, :, np.newaxis]
    #display_image(foreground)
    pixels_fg = np.array([[normalize_rgb(pixel) for pixel in row] for row in foreground])
    
    fg_mask = np.where(mask == 3, 0, 1).astype('uint8')
    background = RGB_img * fg_mask[:, :, np.newaxis]
    #display_image(background)
    pixels_bg = np.array([[normalize_rgb(pixel) for pixel in row] for row in background])

    if len(np.unique(pixels_bg)) > len(np.unique(pixels_fg)):
        pixels_fg, pixels_bg = pixels_bg, pixels_fg
        fg_mask, bg_mask = bg_mask, fg_mask

    posterized_fg = posterize(pixels_fg, num_colors)
    #display_image(posterized_fg)

    posterized_bg = posterize(pixels_bg, 4)
    #display_image(posterized_bg)

    outlines = apply_outlines(filename)
    #display_image(outlines)

    composite = np.zeros(RGB_img.shape)
    outline_colors = []

    for row in range(0, RGB_img.shape[0]):
        for col in range(0, RGB_img.shape[1]):
            if outlines[row][col] == 255:
                outline_colors.append(composite[row][col])
            if bg_mask[row][col] == 0:
                if (row%2 == 0 and col%2 == 0) or (row%2 == 1 and col%2 == 1):
                    composite[row][col] = posterized_bg[row][col]
                else:
                    composite[row][col] = bg_color
            else:
                composite[row][col] = posterized_fg[row][col]

    outline_color = scipy.stats.mode(outline_colors)[0]
    for row in range(0, RGB_img.shape[0]):
        for col in range(0, RGB_img.shape[1]):
            if outlines[row][col] == 255:
                composite[row][col] = outline_color

    #display_image(composite)
    return composite
    

def compute_gradient_map(image):
    greyscale = np.dot(image, [.33, .33, .33])
    derivative_x = scipy.ndimage.sobel(greyscale, 0)
    derivative_y = scipy.ndimage.sobel(greyscale, 1)
    gradient_map = np.hypot(derivative_x, derivative_y)
    gradient_map *= 255.0 / np.max(gradient_map)

    gradient_map_3d = np.zeros(image.shape)
    for row in range(0, image.shape[0]):
        for col in range(0, image.shape[1]):
            gradient_map_3d[row][col] = (gradient_map[row][col], gradient_map[row][col], gradient_map[row][col])

    return gradient_map_3d.astype('uint8')


def display_image(image):
    plt.imshow(image)
    plt.show()


def save_image(image, filename):
    plt.imsave(filename, image)


def get_pixels(filename):
    BGR_img = cv2.imread(filename)
    RGB_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
    pixels = np.array([[normalize_rgb(pixel) for pixel in row] for row in RGB_img])
    return pixels


def get_colors(floss_colors):
    rgb = zip(*floss_colors)[2]
    colors = np.array([normalize_rgb(pixel) for pixel in rgb])
    return colors


def normalize_rgb(rgb_pixel):
    (red_val, green_val, blue_val) = rgb_pixel
    return [red_val / 256., green_val / 256., blue_val / 256.]


def posterize(pixels, num_colors):
    flattened_img = np.array([pixel for row in pixels for pixel in row])
    mean_colors = run_k_means(flattened_img, num_colors, 7)
    colors = get_colors(floss_colors)
    selected_colors = convert_to_floss(mean_colors, colors)
    recolored_img = np.zeros(pixels.shape)

    for row, row_data in enumerate(pixels):
        for col, rgb in enumerate(row_data):
            i = closest_color(rgb, selected_colors)
            recolored_img[row][col] = selected_colors[i]

    return recolored_img


def convert_to_floss(mean_colors, colors):
    selected_colors = np.zeros((len(mean_colors), 3))
    for index, rgb in enumerate(mean_colors):
        i = closest_color(rgb, colors)
        selected_colors[index] = colors[i]
    return selected_colors


def run_k_means(pixels, num_colors, iterations=10):
    np.random.shuffle(pixels)
    i = 0
    s = 0
    means = np.zeros((num_colors, 3))
    while s < num_colors:
        vals = pixels[i]
        if (vals.all() not in means):
            means[s] = pixels[i]
            s += 1
        i += 1
    for i in range(iterations):
        means = compute_means(pixels, num_colors, means)
        #print str(i) + ": " + str(means)
    return means


def compute_means(pixels, num_colors, old_means):
    indexes = [closest_color(pixel, old_means) for pixel in pixels]
    means = []
    for j in range(num_colors):
        points = []
        for pixel, i in zip(pixels, indexes):
            if i == j:
                points.append(pixel)
        means.append(np.mean(points, axis=0))
    return means


def closest_color(pixel, means):
    return min(range(len(means)), key=lambda i: distance(pixel, means[i]))


def distance(p, q):
    return sum((p_i - q_i) ** 2 for p_i, q_i in zip(p, q))


def apply_outlines(filename):
    greyscale_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    median = np.median(greyscale_image)
    sigma = 0.5
    lower = int(max(0, (1.0 - sigma) * median * 3))
    upper = int(min(500, (1.0 + sigma) * median * 3))
    outlines = cv2.Canny(greyscale_image, lower, upper)
    return outlines


def main():
    usage = "usage: %prog -f -o -n -c \n" 
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--filename", dest="filename", help="the input filename")
    parser.add_option("-o", "--output", dest="output", help="the output filename")
    parser.add_option("-n", "--num_colors", dest="num_colors", help="the number of colors")
    parser.add_option("-c", "--canvas_color", dest="canvas_color", help="the color of the canvas (or background): can be 1 or 0")
    (options, args) = parser.parse_args()
    
    pattern = create_pattern(options.filename, int(options.num_colors), int(options.canvas_color))
    save_image(pattern, options.output)


if __name__ == "__main__":
    main()

