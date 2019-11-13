import argparse
import cv2
import numpy as np
import sys

from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import match_template

# Parsing arguments
parser = argparse.ArgumentParser(description="Run code for homework2")
parser.add_argument("-p", "--process", default=False,
                    help="specified if want to process video into frames")
parser.add_argument("-g", "--gray", default=False,
                    help="specified to produce grayscale frames")
parser.add_argument("-n", "--ncc", default=False,
                    help="specified to produce normalized cross correlation coefficients")
parser.add_argument("video_path", nargs="?", default="video.MOV",
                    help="a path to video file to convert into frames")
args = parser.parse_args()

dir_path = "../pictures/"
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
                cv2.imwrite(dir_path + "grayscale_frame_" + str(count) + ".jpg", gray)
            else:
                cv2.imwrite(dir_path + "frame_" + str(count) + ".jpg", frame)
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
    plt.show(block=False)


# Plot pixel shifts of the maximum value of the normalized cross correlation
# matrix among all the frames.
def plot_pixel_shifts(path, template_path):
    template = cv2.imread(template_path, 0)
    max_coordinates = []
    
    print("gathering coordinates..")
    for i in range(frame_count):
        img = cv2.imread(path + "frame_" + str(i) + ".jpg", 0)
        result = match_template(img, template, pad_input=True)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1] # Get the coordinate of the max value
        max_coordinates.append([x, y])

    # Get template's name
    name = template_path.split("/")[2].split(".")[0]
    np.save("ncc_" + name + ".npy", max_coordinates)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1, ylabel="X pixel shift", xlabel="Y pixel shift")
    xs = [p[0]for p in max_coordinates]
    ys = [p[1] for p in max_coordinates]
    ax.plot(xs, ys)
    plt.show(block=False)

    return max_coordinates


# Synthesize an image with a synthetic aperature
def synthesize(matrix, template_name):
    first = matrix[0]
    all_translated_images = 0

    print("synthesizing..")
    for i in range(frame_count):
        img = cv2.imread(dir_path + "frame_" + str(i) + ".jpg", cv2.IMREAD_COLOR)
        current = matrix[i]
        frame_shift_x = -1 * (current[0] - first[0])
        frame_shift_y = -1 * (current[1] - first[1])
        translation_matrix = np.array([[1, 0, frame_shift_x], [0, 1, frame_shift_y]], np.float32)
        translated_img = cv2.warpAffine(img, translation_matrix, (1920, 1080))
        translated_img = translated_img.astype(np.int32) # deal with overflow
        all_translated_images += translated_img

    all_translated_images = np.true_divide(all_translated_images, frame_count)
    all_translated_images = all_translated_images.astype(np.uint8)
    cv2.imwrite("../pictures/synthesized_image_" + template_name + ".jpg", all_translated_images)


if __name__ == "__main__":
    if args.process:
        readvideo(args.video_path)

    # 2.3 - 2.5
    template_ncc(dir_path + "frame_0.jpg", dir_path + "boo.jpg")
    if args.ncc:
        ncc_template1 = plot_pixel_shifts(dir_path, dir_path + "boo.jpg")
    else:
        ncc_template1 = np.load("ncc_boo.npy")
    synthesize(ncc_template1, "dog")

    # 2.6
    if args.ncc:
        ncc_template2 = plot_pixel_shifts(dir_path, dir_path + "can.jpg")
    else:
        ncc_template2 = np.load("ncc_can.npy")
    synthesize(ncc_template2, "can")
