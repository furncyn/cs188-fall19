import cv2
import sys


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

if __name__ == "__main__":
    readvideo(sys.argv[1])
