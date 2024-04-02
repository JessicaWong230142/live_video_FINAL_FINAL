import cv2
import numpy as np
from turn_prediction import turn_prediction
#cited code from https://www.labellerr.com/blog/real-time-lane-detection-for-self-driving-cars-using-opencv/

#detects lane lines and midline by taking in frame of video
def detect_lines(frame):
    midpoint1 = None
    midpoint2 = None

    #finds height and width from frame
    height, width = frame.shape[:2]

    #calculates coordinates of rectangle mask using frame height and width
    mask_size = max(height, width) // 2
    mask_left = int((width - mask_size) / 2) - 140
    mask_top = int((height - mask_size) / 2) - 90
    mask_right = mask_left + mask_size + 180
    mask_bottom = mask_top + mask_size + 100

    #sets rectangle mask from frame
    mask = frame[mask_top:mask_bottom, mask_left:mask_right]

    #preprocesses frame with image processing: convert contents of rectangle mask to grayscale and finds edges in the mask
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
    equalized = cv2.equalizeHist(adaptive_threshold)
    edges = cv2.Canny(equalized, 30, 100, apertureSize=3)

    #detects lines from canny edge detector
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=50, maxLineGap=999)

    #if lines are detected
    if lines is not None:
        # Lists to store endpoints of detected lines
        left_lane_all_lines = []
        right_lane_all_lines = []

        #loops through each line detected
        for line in lines:
            x1, y1, x2, y2 = line[0]

            #calculate the slope of the line
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

            #determine whether the line is on the left lane or the right lane
            if slope < -0.3:
                left_lane_all_lines.append((x1, y1))
                left_lane_all_lines.append((x2, y2))
            elif slope > 0.3:
                right_lane_all_lines.append((x1, y1))
                right_lane_all_lines.append((x2, y2))

        #find a single line for left line and right line
        if left_lane_all_lines and right_lane_all_lines:
            left_lane_line = cv2.fitLine(np.array(left_lane_all_lines), cv2.DIST_L2, 0, 0.01, 0.01)
            left_slope = left_lane_line[1] / left_lane_line[0]
            left_intercept = left_lane_line[3] - left_slope * left_lane_line[2]

            #calculate x coords of left line in mask
            left_x1 = max(0, min(int((mask_bottom - left_intercept) / left_slope), width))
            left_x2 = max(0, min(int((mask_top - left_intercept) / left_slope), width))

            #draw left lane line
            cv2.line(frame, (left_x1, mask_bottom), (left_x2, mask_top), (0, 0, 255), 3)

            right_lane_line = cv2.fitLine(np.array(right_lane_all_lines), cv2.DIST_L2, 0, 0.01, 0.01)
            right_slope = right_lane_line[1] / right_lane_line[0]
            right_intercept = right_lane_line[3] - right_slope * right_lane_line[2]

            #calculate x coords of right line in mask
            right_x1 = max(0, min(int((mask_bottom - right_intercept) / right_slope), width))
            right_x2 = max(0, min(int((mask_top - right_intercept) / right_slope), width))

            #draw right lane line
            cv2.line(frame, (right_x1, mask_bottom), (right_x2, mask_top), (0, 0, 255), 3)

            #find midpts of left and right lane line
            left_midpoint = ((left_x1 + left_x2) // 2, (mask_top + mask_bottom) // 2)
            right_midpoint = ((right_x1 + right_x2) // 2, (mask_top + mask_bottom) // 2)

            #find midpt of midline
            midline_midpoint = ((left_midpoint[0] + right_midpoint[0]) // 2, (left_midpoint[1] + right_midpoint[1]) // 2)

            #draw parallel midline to left and right lane line
            cv2.line(frame, (midline_midpoint[0], mask_bottom), (midline_midpoint[0], mask_top), (0, 255, 0), 2)

    #draws rectangle around rectangle mask
    cv2.rectangle(frame, (mask_left, mask_top), (mask_right, mask_bottom), (255, 255, 255), 2)

    #returns frame with lines and rectangle mask drawn
    return frame

#takes in video feed from camera
video_feed = cv2.VideoCapture('apple.mov')

#cited code from https://www.youtube.com/watch?v=drp_mr2x6A8

#continuosly capture frames from video_feed
while True:

    #reads frames
    ret, frame = video_feed.read()

    #coordinates of trapezoid in original video
    tl = (750, 800)
    bl = (480, 1000)
    tr = (1190, 800)
    br = (1400, 1000)

    #sets points up for perspective transform
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [500, 0], [500, 480]])

    #apply perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (650, 700))
    find_lines = detect_lines(result)

    #put perspective transform onto frame with detected lines
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    lines_on_original = cv2.warpPerspective(find_lines, inv_matrix, (frame.shape[1], frame.shape[0]))

    turn_predicted = turn_prediction(lines_on_original)
    #overlay frame with detected lines onto original frame
    result_frame = cv2.addWeighted(frame, 0.5, turn_predicted, 0.5, 0)
    cv2.imshow('result', result_frame)

    #ends program if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releases video_feed and closes opencv windows
video_feed.release()
cv2.destroyAllWindows()
