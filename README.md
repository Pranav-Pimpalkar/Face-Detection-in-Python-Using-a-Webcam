# Face-Detection-in-Python-Using-a-Webcam
Face Detection in Python Using a Webcam
If you’re looking for an introduction to face detection, then you’ll want to read Traditional Face Detection With Python before diving into this tutorial. Once you’ve gotten a solid understanding of how to detect faces with Python, you can move from detecting faces in images to detecting them in video via a webcam, which is exactly what you’ll explore below.

Before you ask any questions in the comments section:

Do not skip over the blog post and try to run the code. You must understand what the code does not only to run it properly but to troubleshoot it as well.
Make sure to use OpenCV v2.
You need a working webcam for this script to work properly.
Review the other comments/questions as your questions have probably already been addressed.
Thank you.

Free Bonus: Click here to get the Python Face Detection & OpenCV Examples Mini-Guide that shows you practical code examples of real-world Python computer vision techniques.

Note: If you’d like to tackle face recognition instead of face detection, then Build Your Own Face Recognition Tool With Python has you covered.

Pre-requisites
OpenCV installed (see the previous blog post for details)
A working webcam

Remove ads
The Code
Let’s dive straight into the code, taken from this repository.

import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
Now let’s break it down…

import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
This should be familiar to you. We are creating a face cascade, as we did in the image example.

video_capture = cv2.VideoCapture(0)
This line sets the video source to the default webcam, which OpenCV can easily capture.

NOTE: You can also provide a filename here, and Python will read in the video file. However, you need to have ffmpeg installed for that since OpenCV itself cannot decode compressed video. Ffmpeg acts as the front end for OpenCV, and, ideally, it should be compiled directly into OpenCV. This is not easy to do, especially on Windows.

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
Here, we capture the video. The read() function reads one frame from the video source, which in this example is the webcam. This returns:

The actual video frame read (one frame on each loop)
A return code
The return code tells us if we have run out of frames, which will happen if we are reading from a file. This doesn’t matter when reading from the webcam, since we can record forever, so we will ignore it.

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
Again, this code should be familiar. We are merely searching for the face in our captured frame.

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
We wait for the ‘q’ key to be pressed. If it is, we exit the script.

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
Here, we are just cleaning up.

Test!
