# Photos to Needlework Designs

# Introduction
This project takes a photo as input, and creates a cross-stitch needlework pattern as output with the specified number of colors and canvas background color. I computationally found the best set of colors, and ensured that detail is not lost by applying well-defined edges, vivid colors, high detail in the foreground, and subtle backgrounds.

# Language
Python 2.7

# Packages
The following libraries/packages are needed:
- cv2
- random
- scipy.stats
- scipy.ndimage
- numpy
- matplotlib.pyplot

# How to Use

--Command Line Instructions--

To generate the pattern, type:
python cross_stitch_pattern.py -f [JPG input image filename] -o [output image filename] -n [number of colors] -c [canvas color (0 for black, 1 for white)]

For example, the following command will create a pattern with 24 colors on a white canvas:
python cross_stitch_pattern.py -f images/birds_input.jpg -o images/birds_output.jpg -n 24 -c 1

Note that a 600x400 image can take about 10 minutes to process 20 colors. 
