import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 30
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                           minLineLength, maxLineGap)

# def average_slope_intercept(lines):
#     left_lines, right_lines = [], []
#     left_weights, right_weights = [], []
#     if lines is None:
#         return None, None

#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             if x1 == x2:
#                 continue
#             slope = (y2 - y1) / (x2 - x1)
#             intercept = y1 - slope * x1
#             # length = np.sqrt((y2 - y1)*2 + (x2 - x1)*2)
#             length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#             if slope < 0:
#                 left_lines.append((slope, intercept))
#                 left_weights.append(length)
#             else:
#                 right_lines.append((slope, intercept))
#                 right_weights.append(length)

#     left_lane  = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
#     right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
#     return left_lane, right_lane
def average_slope_intercept(lines):
    left_lines, right_lines = [], []
    left_weights, right_weights = [], []
    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Skip vertical lines
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Skip invalid or near-horizontal lines
            if np.isnan(slope) or np.isnan(intercept) or abs(slope) < 0.3:
                continue

            # FIXED length formula ✅
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane  = (np.dot(left_weights, left_lines) / np.sum(left_weights)) if left_weights else None
    right_lane = (np.dot(right_weights, right_lines) / np.sum(right_weights)) if right_weights else None
    return left_lane, right_lane

# def pixel_points(y1, y2, line):
#     if line is None:
#         return None
#     slope, intercept = line
#     x1 = int((y1 - intercept) / slope)
#     x2 = int((y2 - intercept) / slope)
#     return ((x1, int(y1)), (x2, int(y2)))
def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line

    if slope == 0 or np.isnan(slope) or np.isinf(slope):
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # Handle if x is nan or inf
    if np.isnan(x1) or np.isnan(x2) or np.isinf(x1) or np.isinf(x2):
        return None

    return ((x1, int(y1)), (x2, int(y2)))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=10):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 0.8, line_image, 1.2, 0.0)

def frame_processor(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(frame, lane_lines(frame, hough))
    return result


VIDEO_SOURCE = "input.mp4"  

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Error: Cannot open video source.")
else:
    print("✅ Video source opened successfully.")

root = tk.Tk()
root.title("Real-Time Lane Detection")
root.geometry("900x700+100+10")

label = tk.Label(root)
label.pack()

def show_frame():
    if not cap.isOpened():
        print("⚠ Video stream closed.")
        return


    ret, frame = cap.read()
    # if not ret:
    #     print("⚠ No frame received. End of video or camera not ready.")
    #     root.after(1000, show_frame)
    #     return
    if not ret:
        print("⚠ Restarting video...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start
        root.after(30, show_frame)
        return

    frame = cv2.resize(frame, (800, 600))
    processed = frame_processor(frame)

    img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, show_frame)

exit_btn = tk.Button(root, text="Quit", fg="red", command=lambda: (root.destroy(), cap.release()))
exit_btn.pack(side=tk.BOTTOM, pady=10)

show_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()