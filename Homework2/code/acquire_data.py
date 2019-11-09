import cv2
import sys

cap = cv2.VideoCapture(sys.argv[1])
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