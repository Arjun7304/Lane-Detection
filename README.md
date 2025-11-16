🚗 Lane Detection System (OpenCV + Tkinter GUI)

This project implements a real-time lane detection system using OpenCV, NumPy, and Tkinter for GUI display.
It detects left and right lane boundaries from a video stream using edge detection, region masking, and Hough Line Transform.

✨ Features

✔ Real-time lane detection
✔ GUI window using Tkinter
✔ Region of Interest (ROI) filtering
✔ Canny edge detection
✔ Hough Line Transform for line extraction
✔ Weighted averaging of lane lines
✔ Smooth lane overlay on video frames
✔ Auto-restart when video ends

🎥 Demo Workflow

Load video (input.mp4)
Convert frame → grayscale
Apply Gaussian blur
Detect edges using Canny
Apply region mask
Detect lane lines (Hough Transform)
Average left/right lane boundaries
Draw final lane lines on output frame
Show inside GUI window

🛠 Technologies Used

Python
OpenCV
NumPy
Tkinter
Pillow (PIL)

