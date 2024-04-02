import tkinter as tk
from tkinter import *
from PIL import ImageTk
from PIL import Image
import cv2
import numpy as np
from lane_detection import detect_lines
from turn_prediction import turn_prediction

# Function to update camera feed with overlay
def update_overlay_feed(top_left_frame):
    try:
        url = 'apple.mov'
        cap = cv2.VideoCapture(url)

        # Function to convert OpenCV image to Tkinter PhotoImage
        def convert_to_photo_image(frame):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            return photo

        # Function to update the label within top_left_frame with a new image
        def update_label():
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame,(650,380))
                photo = convert_to_photo_image(frame)
                camera_feed_label.configure(image=photo)
                camera_feed_label.image = photo
                top_left_frame.after(10, update_label)  # Update every 10 milliseconds
            else:
                print("Error reading frame from camera feed")

        # Create a label within top_left_frame for displaying the camera feed
        camera_feed_label = tk.Label(top_left_frame)
        camera_feed_label.pack()

        # Start updating the label
        update_label()

    except Exception as e:
        print(f"Error updating camera feed: {e}")

#Function to display camera feed without overlay
def update_camera_feed(bottom_left_frame):
    try:
        url = 'apple.mov'
        cap = cv2.VideoCapture(url)

        # Function to convert OpenCV image to Tkinter PhotoImage
        def convert_to_photo_image(frame):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            return photo

        # Function to update the label within top_left_frame with a new image
        def update_label():
            ret, frame = cap.read()

            if ret:
                # coordinates of trapezoid in original video
                tl = (750, 800)
                bl = (480, 1000)
                tr = (1190, 800)
                br = (1400, 1000)

                # sets points up for perspective transform
                pts1 = np.float32([tl, bl, tr, br])
                pts2 = np.float32([[0, 0], [0, 480], [500, 0], [500, 480]])

                # apply perspective transform
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                result = cv2.warpPerspective(frame, matrix, (650, 700))
                find_lines = detect_lines(result)

                # put perspective transform onto frame with detected lines
                inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
                lines_on_original = cv2.warpPerspective(find_lines, inv_matrix, (frame.shape[1], frame.shape[0]))

                turn_predicted = turn_prediction(lines_on_original)
                # overlay frame with detected lines onto original frame
                result_frame = cv2.addWeighted(frame, 0.5, turn_predicted, 0.5, 0)
                final_frame = cv2.resize(result_frame, (650, 380))
                photo = convert_to_photo_image(final_frame)
                camera_feed_label.configure(image=photo)
                camera_feed_label.image = photo
                bottom_left_frame.after(10, update_label)  # Update every 10 milliseconds
            else:
                print("Error reading frame from camera feed")

        # Create a label within top_left_frame for displaying the camera feed
        camera_feed_label = tk.Label(bottom_left_frame)
        camera_feed_label.pack()

        # Start updating the label
        update_label()

    except Exception as e:
        print(f"Error updating camera feed: {e}")
