import argparse
import cv2
import numpy as np
import sys

from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import match_template

# Parsing arguments
parser = argparse.ArgumentParser(description="Run code for homework2")
parser.add_argument("-p","--process",default=False,
                    help="specified if want to process video into frames")
parser.add_argument("-g","--gray",default=False,
                    help="specified to produce grayscale frames")
parser.add_argument("video_path",nargs="?",default="video.MOV",
                    help="a path to video file to convert into frames")
args = parser.parse_args()

dir_path = "../pictures/"
boo_template_path = "../pictures/boo.jpg"
frame_count = 280 # known constant


# Read video file and convert the video into a sequence of frames
def readvideo(path):
    cap = cv2.VideoCapture(path)
    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Convert to grayscale, then save each frame
            print("Processing frame " + str(count))
            if args.gray:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("../pictures/grayscale_frame_" + str(count) + ".jpg", gray)
            else:
                cv2.imwrite("../pictures/frame_" + str(count) + ".jpg", frame)
            count = count + 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# Calculate the Normalized Cross Correlation of the template
def template_ncc(img_path, template_path):
    img = cv2.imread(img_path, 0)
    template = cv2.imread(template_path, 0)
    result = match_template(img, template, pad_input=True)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1,
        ylabel="Pixel location in Y direction",
        xlabel="Pixel location in X direction"
    )
    ax.imshow(result, cmap=plt.cm.gray)
    ax.autoscale(False)
    plt.show()


# Plot pixel shifts of the maximum value of the normalized cross correlation
# matrix among all the frames.
def plot_pixel_shifts(path, template_path):
    template = cv2.imread(template_path, 0)
    max_coordinates = []

    for i in range(frame_count):
        img = cv2.imread(path + "frame_" + str(i) + ".jpg", 0)
        result = match_template(img, template, pad_input=True)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1] # Get the coordinate of the max value
        max_coordinates.append([x, y])

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, ylabel="X pixel shift", xlabel="Y pixel shift")
    xs = [p[0]for p in max_coordinates]
    ys = [p[1] for p in max_coordinates]
    ax.plot(xs, ys)
    plt.show(block=True)


if __name__ == "__main__":
    if args.process:
        readvideo(args.video_path)
    template_ncc(dir_path + "frame_0.jpg", boo_template_path)
    plot_pixel_shifts(dir_path, boo_template_path)
