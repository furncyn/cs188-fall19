import cv2


# Convert vdo into sequence of frames
# The function takes in a path to the video file
def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1

    while success:
        success, image = vidObj.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
        # cv2.imshow("Frame", gray_image) # For testing only
        cv2.imwrite("frame%d.jpg" % count, gray_image) # Save the frame
        count += 1
