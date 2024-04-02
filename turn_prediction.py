import cv2

def turn_prediction(frame):
    text_forward = "forward"
    text_left = "turning left"
    text_right = "turning right"
    text_position = (100, 100)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    thickness = 3
    line_type = cv2.LINE_AA

    cv2.putText(frame, text_forward, text_position, fontFace, font_scale, font_color, thickness, line_type)

    #setting images as variables
    template_left_path = 'left_arrow_template.jpg'
    template_right_path = 'right_turn_arrow.jpg'

    template_left = cv2.imread(template_left_path, cv2.IMREAD_GRAYSCALE)
    template_right = cv2.imread(template_right_path, cv2.IMREAD_GRAYSCALE)

    # Define the threshold for template matching
    threshold = 0.8

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform template matching with left arrow template
    result_left = cv2.matchTemplate(gray_frame, template_left, cv2.TM_CCOEFF_NORMED)
    min_val_left, max_val_left, min_loc_left, max_loc_left = cv2.minMaxLoc(result_left)

    # Perform template matching with right arrow template
    result_right = cv2.matchTemplate(gray_frame, template_right, cv2.TM_CCOEFF_NORMED)
    min_val_right, max_val_right, min_loc_right, max_loc_right = cv2.minMaxLoc(result_right)

    # Check if the maximum correlation value is above the threshold for left arrow
    if max_val_left > threshold:
        # Get the coordinates of the left arrow bounding box
        left_top_left = max_loc_left
        left_bottom_right = (left_top_left[0] + template_left.shape[1], left_top_left[1] + template_left.shape[0])

        # Draw a bounding box around the left arrow
        cv2.rectangle(frame, left_top_left, left_bottom_right, (0, 255, 0), 2)

        # Print message indicating left turn
        cv2.putText(frame, text_left, text_position, fontFace, font_scale, font_color, thickness, line_type)
        print("turning left")

    # Check if the maximum correlation value is above the threshold for right arrow
    if max_val_right > threshold:
        # Get the coordinates of the right arrow bounding box
        right_top_left = max_loc_right
        right_bottom_right = (right_top_left[0] + template_right.shape[1], right_top_left[1] + template_right.shape[0])

        # Draw a bounding box around the right arrow
        cv2.rectangle(frame, right_top_left, right_bottom_right, (0, 255, 0), 2)

        # Print message indicating right turn
        cv2.putText(frame, text_right, text_position, fontFace, font_scale, font_color, thickness, line_type)
        print("turning right")

    # Return the frame with turn prediction overlay
    return frame
