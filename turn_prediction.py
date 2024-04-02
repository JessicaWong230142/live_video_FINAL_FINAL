import cv2

def turn_prediction(frame):

    #text to put onto the screen to rep forward, left, right
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
    left_template_img = 'left_arrow_template.jpg'
    right_template_img = 'right_turn_arrow.jpg'

    #read both of the images
    template_left = cv2.imread(left_template_img, cv2.IMREAD_GRAYSCALE)
    template_right = cv2.imread(right_template_img, cv2.IMREAD_GRAYSCALE)

    #define a threshold for the image matching
    threshold = 0.8

    #change frame to grayscale
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #template matching with left arrow img
    result_left = cv2.matchTemplate(gray_scale, template_left, cv2.TM_CCOEFF_NORMED)
    min_val_left, max_val_left, min_loc_left, max_loc_left = cv2.minMaxLoc(result_left)

    #template matching with left arrow img
    result_right = cv2.matchTemplate(gray_scale, template_right, cv2.TM_CCOEFF_NORMED)
    min_val_right, max_val_right, min_loc_right, max_loc_right = cv2.minMaxLoc(result_right)

    #compare ag threshold
    if max_val_left > threshold:
        left_top_left = max_loc_left
        left_bottom_right = (left_top_left[0] + template_left.shape[1], left_top_left[1] + template_left.shape[0])

        #draw box around detected left arrow
        cv2.rectangle(frame, left_top_left, left_bottom_right, (0, 255, 0), 2)

        #print turning left
        cv2.putText(frame, text_left, text_position, fontFace, font_scale, font_color, thickness, line_type)
        print("turning left")

    #compare ag threshold
    if max_val_right > threshold:
        # Get the coordinates of the right arrow bounding box
        right_top_left = max_loc_right
        right_bottom_right = (right_top_left[0] + template_right.shape[1], right_top_left[1] + template_right.shape[0])

        #draw box around detected right arrow
        cv2.rectangle(frame, right_top_left, right_bottom_right, (0, 255, 0), 2)

        #print turning right
        cv2.putText(frame, text_right, text_position, fontFace, font_scale, font_color, thickness, line_type)
        print("turning right")

    #return frame
    return frame
