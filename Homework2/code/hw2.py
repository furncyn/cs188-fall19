import argparse
import cv2
import numpy as np
import sys

from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import match_template

rabbit_template_path = "../pictures/rabbit.jpg"


parser = argparse.ArgumentParser(description="Run code for homework2")
parser.add_argument("video_path",nargs="?",default="video.mp4",
                    help="a path to video file to convert into frames")
args = parser.parse_args()

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("../pictures/grayscale_frame_" + str(count) + ".jpg", gray)
            count = count + 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# Calculate the Normalized Cross Correlation of the template
def detect_template(img_path, template_path):
    img = cv2.imread(img_path, 0)
    template = cv2.imread(template_path, 0)
    result = match_template(img, template, pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1] # Top left coordinate of the matching pattern is returned

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1,
        ylabel="Pixel location in Y direction",
        xlabel="Pixel location in X direction"
    )

    ax.imshow(result, cmap=plt.cm.gray)
    ax.autoscale(False)
    print(result)
    plt.show()


if __name__ == "__main__":
    # readvideo(args.video_path)
    detect_template("../pictures/frame_0.jpg", rabbit_template_path)
