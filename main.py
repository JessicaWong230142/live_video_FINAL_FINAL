# there is cited code in the lane_detection.py file!
import cv2
import numpy as np
import os
from GUI import window
from lane_detection import detect_lines
from turn_prediction import turn_prediction

file_path = "USER.db"

if not os.path.exists(file_path):
    open(file_path, 'w').close()

window() #call window function

# #takes in video feed from camera
# video_feed = cv2.VideoCapture('apple.mov')
#
# #continuosly capture frames from video_feed
# while True:
#
#     #reads frames
#     ret, frame = video_feed.read()
#
#     #coordinates of trapezoid in original video
#     tl = (750, 800)
#     bl = (480, 1000)
#     tr = (1190, 800)
#     br = (1400, 1000)
#
#     #sets points up for perspective transform
#     pts1 = np.float32([tl, bl, tr, br])
#     pts2 = np.float32([[0, 0], [0, 480], [500, 0], [500, 480]])
#
#     #apply perspective transform
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     result = cv2.warpPerspective(frame, matrix, (650, 700))
#     find_lines = detect_lines(result)
#
#     #put perspective transform onto frame with detected lines
#     inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
#     lines_on_original = cv2.warpPerspective(find_lines, inv_matrix, (frame.shape[1], frame.shape[0]))
#
#     turn_predicted = turn_prediction(lines_on_original)
#     #overlay frame with detected lines onto original frame
#     result_frame = cv2.addWeighted(frame, 0.5, turn_predicted, 0.5, 0)
#     cv2.imshow('result', result_frame)
#
#     #ends program if q is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# #releases video_feed and closes opencv windows
# video_feed.release()
# cv2.destroyAllWindows()
