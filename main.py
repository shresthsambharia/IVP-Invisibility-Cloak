

import cv2

import numpy as np

import mediapipe as mp

import time

# ============================================================

# WHITE CLOTH INVISIBILITY CLOAK

# OpenCV + MediaPipe Hand Gesture Control

#

# Index finger          : Move cursor

# Thumb + index pinch   : Select button

# + / -                 : Change sensitivity

# B                     : Recapture background

# M                     : Toggle mask preview

# SPACE                 : Toggle cloak

# Q / ESC               : Exit

# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 960

FRAME_HEIGHT = 540

CAPTURE_SECONDS = 3

COLORS = ["RED", "BLUE", "GREEN", "WHITE"]

HSV_RANGES = {

    "RED": [

        (np.array([0, 100, 60]), np.array([12, 255, 255])),

        (np.array([168, 100, 60]), np.array([180, 255, 255]))

    ],

    "BLUE": [

        (np.array([95, 75, 45]), np.array([135, 255, 255]))

    ],

    "GREEN": [

        (np.array([35, 60, 40]), np.array([90, 255, 255]))

    ],

    "WHITE": [

        (np.array([0, 0, 155]), np.array([180, 95, 255]))

    ]

}

COLOR_BGR = {

    "RED": (50, 50, 255),

    "BLUE": (255, 120, 40),

    "GREEN": (50, 220, 50),

    "WHITE": (245, 245, 245)

}

mp_hands = mp.solutions.hands

KERNEL_OPEN = np.ones((3, 3), np.uint8)

KERNEL_CLOSE = np.ones((9, 9), np.uint8)

KERNEL_DILATE = np.ones((5, 5), np.uint8)

# ------------------------------------------------------------

# CAMERA

# ------------------------------------------------------------

def read_frame(camera):

    ok, frame = camera.read()

    if not ok or frame is None:

        return None

    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    return frame

# ------------------------------------------------------------

# BACKGROUND CAPTURE

# ------------------------------------------------------------

def capture_background(camera):

    print("Background capture started.")

    print("Move out of the camera view and keep the camera still.")

    background = None

    start_time = time.time()

    while time.time() - start_time < CAPTURE_SECONDS:

        frame = read_frame(camera)

        if frame is None:

            continue

        background = frame.copy()

        remaining = max(

            0,

            int(CAPTURE_SECONDS - (time.time() - start_time)) + 1

        )

        preview = frame.copy()

        cv2.putText(

            preview,

            f"CAPTURING BACKGROUND: {remaining}",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (0, 255, 255),

            2,

            cv2.LINE_AA

        )

        cv2.putText(

            preview,

            "Keep the scene empty and camera stationary",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2,

            cv2.LINE_AA

        )

        cv2.imshow("Invisibility Cloak", preview)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):

            return None

    if background is not None:

        # Light smoothing reduces sensor noise.

        background = cv2.GaussianBlur(

            background, (5, 5), 0

        )

    print("Background captured successfully.")

    return background

# ------------------------------------------------------------

# CLOTH DETECTION

# ------------------------------------------------------------

def detect_cloth(frame, color_name, sensitivity, background):

    """

    Detect the selected cloth color and suppress static

    background regions using background subtraction.

    sensitivity:

        0   = stricter foreground/color filtering

        100 = more permissive detection

    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Detect the selected color in HSV space.

    color_mask = np.zeros(

        frame.shape[:2], dtype=np.uint8

    )

    for lower, upper in HSV_RANGES[color_name]:

        part = cv2.inRange(hsv, lower, upper)

        color_mask = cv2.bitwise_or(color_mask, part)

    saturation = hsv[:, :, 1]

    if color_name == "WHITE":

        # White/light fabric should have low saturation.

        max_saturation = max(35, 95 + sensitivity // 5)

        color_mask[saturation > max_saturation] = 0

    else:

        # Colored fabric should have sufficient saturation.

        min_saturation = max(25, 115 - sensitivity)

        color_mask[saturation < min_saturation] = 0

    # 2. Foreground detection relative to captured background.

    if background is not None:

        if background.shape != frame.shape:

            background = cv2.resize(

                background,

                (frame.shape[1], frame.shape[0])

            )

        # Blur both images slightly to reduce camera noise.

        current_blurred = cv2.GaussianBlur(

            frame, (5, 5), 0

        )

        background_blurred = cv2.GaussianBlur(

            background, (5, 5), 0

        )

        difference = cv2.absdiff(

            current_blurred,

            background_blurred

        )

        difference_gray = cv2.cvtColor(

            difference, cv2.COLOR_BGR2GRAY

        )

        # Higher sensitivity permits smaller changes.

        diff_threshold = int(

            42 - (sensitivity * 0.20)

        )

        diff_threshold = max(

            20, min(42, diff_threshold)

        )

        _, foreground_mask = cv2.threshold(

            difference_gray,

            diff_threshold,

            255,

            cv2.THRESH_BINARY

        )

        # Fill small gaps in the foreground region.

        foreground_mask = cv2.morphologyEx(

            foreground_mask,

            cv2.MORPH_CLOSE,

            KERNEL_CLOSE

        )

        # Expand slightly to cover cloth boundaries.

        foreground_mask = cv2.dilate(

            foreground_mask,

            KERNEL_DILATE,

            iterations=1

        )

        # Both selected color and foreground must match.

        color_mask = cv2.bitwise_and(

            color_mask,

            foreground_mask

        )

    # 3. Remove isolated noise.

    color_mask = cv2.morphologyEx(

        color_mask,

        cv2.MORPH_OPEN,

        KERNEL_OPEN

    )

    # 4. Fill small holes inside detected cloth.

    color_mask = cv2.morphologyEx(

        color_mask,

        cv2.MORPH_CLOSE,

        KERNEL_CLOSE

    )

    # 5. Remove very small connected components.

    count, labels, stats, _ = cv2.connectedComponentsWithStats(

        color_mask,

        connectivity=8

    )

    cleaned = np.zeros_like(color_mask)

    min_area = int(

        500 + (100 - sensitivity) * 8

    )

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:

            cleaned[labels == i] = 255

    # 6. Feather the boundary for smoother blending.

    soft_mask = cv2.GaussianBlur(

        cleaned, (7, 7), 0

    )

    return soft_mask

# ------------------------------------------------------------

# TEMPORAL STABILIZATION

# ------------------------------------------------------------

def stabilize_mask(current_mask, previous_mask):

    """

    Blend the current mask with the previous frame to reduce

    rapid mask flickering.

    """

    if previous_mask is None:

        return current_mask

    stable = cv2.addWeighted(

        current_mask, 0.70,

        previous_mask, 0.30,

        0

    )

    return stable

# ------------------------------------------------------------

# INVISIBILITY EFFECT

# ------------------------------------------------------------

def apply_cloak(frame, background, mask):

    """

    Replace detected cloth pixels with the corresponding

    captured background pixels using a soft alpha mask.

    """

    if background is None:

        return frame.copy()

    if background.shape != frame.shape:

        background = cv2.resize(

            background,

            (frame.shape[1], frame.shape[0])

        )

    alpha = mask.astype(np.float32) / 255.0

    alpha = alpha[:, :, None]

    foreground = frame.astype(np.float32)

    background_float = background.astype(np.float32)

    output = (

        background_float * alpha

        + foreground * (1.0 - alpha)

    )

    return np.clip(

        output, 0, 255

    ).astype(np.uint8)

# ------------------------------------------------------------

# INTERFACE BUTTONS

# ------------------------------------------------------------

def draw_button(

    frame, rect, label, color,

    active=False, hovered=False

):

    x1, y1, x2, y2 = rect

    if active:

        fill = (40, 125, 65)

    elif hovered:

        fill = (80, 80, 100)

    else:

        fill = (35, 35, 45)

    cv2.rectangle(

        frame, (x1, y1), (x2, y2), fill, -1

    )

    cv2.rectangle(

        frame, (x1, y1), (x2, y2), color, 2

    )

    font_scale = 0.48 if len(label) > 8 else 0.55

    cv2.putText(

        frame,

        label,

        (x1 + 7, y1 + 28),

        cv2.FONT_HERSHEY_SIMPLEX,

        font_scale,

        (255, 255, 255),

        1,

        cv2.LINE_AA

    )

def make_buttons():

    return [

        {

            "name": "RED",

            "rect": (10, 45, 87, 85)

        },

        {

            "name": "BLUE",

            "rect": (94, 45, 171, 85)

        },

        {

            "name": "GREEN",

            "rect": (178, 45, 255, 85)

        },

        {

            "name": "WHITE",

            "rect": (262, 45, 339, 85)

        },

        {

            "name": "BACKGROUND",

            "rect": (350, 45, 475, 85)

        },

        {

            "name": "CLOAK ON/OFF",

            "rect": (486, 45, 620, 85)

        },

        {

            "name": "MASK",

            "rect": (631, 45, 715, 85)

        }

    ]

# ------------------------------------------------------------

# HAND GESTURE TRACKING

# ------------------------------------------------------------

def get_hand_cursor(hand_landmarks, width, height):
    """
    Landmark 8: index fingertip
    Landmark 4: thumb fingertip

    Cursor mapping uses a comfortable hand workspace:
    - Horizontal hand movement maps to the full window.
    - The upper/middle part of the camera view maps to the full
      screen height, so reaching the top control buttons is easier.
    """

    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]

    # Map the index fingertip from a comfortable workspace to the
    # whole application window. Clamp values at both edges.
    x_min, x_max = 0.08, 0.92
    y_min, y_max = 0.12, 0.72

    mapped_x = (index_tip.x - x_min) / (x_max - x_min)
    mapped_y = (index_tip.y - y_min) / (y_max - y_min)

    ix = int(np.clip(mapped_x, 0.0, 1.0) * (width - 1))
    iy = int(np.clip(mapped_y, 0.0, 1.0) * (height - 1))

    # Measure pinch using the real hand landmarks, not the remapped
    # cursor coordinates, so the pinch threshold stays consistent.
    distance = np.hypot(
        (index_tip.x - thumb_tip.x) * width,
        (index_tip.y - thumb_tip.y) * height
    )

    pinch_active = distance < width * 0.045

    tx = int(thumb_tip.x * width)
    ty = int(thumb_tip.y * height)
    tx = max(0, min(width - 1, tx))
    ty = max(0, min(height - 1, ty))

    return (ix, iy), (tx, ty), pinch_active

# ------------------------------------------------------------

# MAIN

# ------------------------------------------------------------

def main():

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():

        print("ERROR: Cannot open webcam.")

        print("Try changing CAMERA_INDEX to 1.")

        return

    camera.set(

        cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH

    )

    camera.set(

        cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT

    )

    background = None

    selected_color = "WHITE"

    sensitivity = 60

    cloak_enabled = True

    show_mask = True

    previous_mask = None

    smooth_cursor = None

    cursor_smoothing = 0.35

    pinch_was_active = False

    last_click_time = 0.0

    buttons = make_buttons()

    print("\n======================================")

    print(" HAND GESTURE INVISIBILITY CLOAK")

    print("======================================")

    print("Index finger       : Move cursor")

    print("Thumb + index pinch: Select a button")

    print("B                  : Capture background")

    print("SPACE              : Toggle cloak")

    print("M                  : Toggle mask window")

    print("+ / -              : Sensitivity")

    print("Q / ESC            : Exit")

    print("======================================")

    try:

        background = capture_background(camera)

        if background is None:

            return

        with mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            model_complexity=0,

            min_detection_confidence=0.60,

            min_tracking_confidence=0.60

        ) as hands:

            while True:

                frame = read_frame(camera)

                if frame is None:

                    print("Camera frame could not be read.")

                    break

                height, width = frame.shape[:2]

                # -----------------------------

                # Detect hand

                # -----------------------------

                rgb = cv2.cvtColor(

                    frame, cv2.COLOR_BGR2RGB

                )

                hand_result = hands.process(rgb)

                cursor = None

                thumb_position = None

                pinch_active = False

                if hand_result.multi_hand_landmarks:

                    hand = hand_result.multi_hand_landmarks[0]

                    raw_cursor, thumb_position, pinch_active = (

                        get_hand_cursor(

                            hand, width, height

                        )

                    )

                    if smooth_cursor is None:

                        smooth_cursor = np.array(

                            raw_cursor, dtype=np.float32

                        )

                    else:

                        smooth_cursor = (

                            cursor_smoothing

                            * np.array(

                                raw_cursor, dtype=np.float32

                            )

                            + (1.0 - cursor_smoothing)

                            * smooth_cursor

                        )

                    cursor = (

                        int(smooth_cursor[0]),

                        int(smooth_cursor[1])

                    )

                # -----------------------------

                # Determine hovered button

                # -----------------------------

                hovered_button = None

                if cursor is not None:

                    cx, cy = cursor

                    for button in buttons:

                        x1, y1, x2, y2 = button["rect"]

                        if (

                            x1 <= cx <= x2

                            and y1 <= cy <= y2

                        ):

                            hovered_button = button["name"]

                            break

                # -----------------------------

                # Handle pinch click

                # -----------------------------

                new_pinch = (

                    pinch_active and not pinch_was_active

                )

                if (

                    new_pinch

                    and hovered_button is not None

                    and time.time() - last_click_time > 0.45

                ):

                    last_click_time = time.time()

                    if hovered_button in COLORS:

                        selected_color = hovered_button

                        previous_mask = None

                        print(

                            "Selected color:",

                            selected_color

                        )

                    elif hovered_button == "BACKGROUND":

                        new_background = capture_background(

                            camera

                        )

                        if new_background is not None:

                            background = new_background

                            previous_mask = None

                        smooth_cursor = None

                    elif hovered_button == "CLOAK ON/OFF":

                        cloak_enabled = not cloak_enabled

                    elif hovered_button == "MASK":

                        show_mask = not show_mask

                pinch_was_active = pinch_active

                # -----------------------------

                # Detect and stabilize cloth mask

                # -----------------------------

                mask = detect_cloth(

                    frame,

                    selected_color,

                    sensitivity,

                    background

                )

                mask = stabilize_mask(

                    mask, previous_mask

                )

                previous_mask = mask.copy()

                # -----------------------------

                # Apply cloak

                # -----------------------------

                if cloak_enabled:

                    output = apply_cloak(

                        frame,

                        background,

                        mask

                    )

                else:

                    output = frame.copy()

                # -----------------------------

                # Draw live control panel

                # -----------------------------

                cv2.rectangle(

                    output,

                    (0, 0),

                    (width, 100),

                    (25, 25, 32),

                    -1

                )

                cv2.putText(

                    output,

                    "HAND GESTURE INVISIBILITY CLOAK",

                    (10, 25),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.60,

                    (0, 255, 0),

                    2,

                    cv2.LINE_AA

                )

                for button in buttons:

                    name = button["name"]

                    if name in COLORS:

                        border_color = COLOR_BGR[name]

                        active = selected_color == name

                    else:

                        border_color = (0, 220, 220)

                        active = (

                            name == "CLOAK ON/OFF"

                            and cloak_enabled

                        ) or (

                            name == "MASK"

                            and show_mask

                        )

                    draw_button(

                        output,

                        button["rect"],

                        name,

                        border_color,

                        active,

                        hovered_button == name

                    )

                # -----------------------------

                # Status and instructions

                # -----------------------------

                status = (

                    f"Color: {selected_color} | "

                    f"Sensitivity: {sensitivity} | "

                    f"Cloak: "

                    f"{'ON' if cloak_enabled else 'OFF'}"

                )

                cv2.putText(

                    output,

                    status,

                    (10, height - 37),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 0),

                    2,

                    cv2.LINE_AA

                )

                cv2.putText(

                    output,

                    "Index finger: Move | Pinch: Select | "

                    "B: Background | SPACE: Cloak | Q: Exit",

                    (10, height - 12),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.42,

                    (255, 255, 255),

                    1,

                    cv2.LINE_AA

                )

                # Cursor feedback

                if cursor is not None:

                    cv2.circle(

                        output,

                        cursor,

                        10,

                        (0, 255, 255),

                        2

                    )

                    if pinch_active:

                        cv2.circle(

                            output,

                            cursor,

                            16,

                            (0, 255, 0),

                            2

                        )

                cv2.imshow(

                    "Invisibility Cloak",

                    output

                )

                if show_mask:

                    cv2.imshow(

                        "Cloth Detection Mask",

                        mask

                    )

                # -----------------------------

                # Keyboard controls

                # -----------------------------

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):

                    break

                elif key == ord("b"):

                    new_background = capture_background(

                        camera

                    )

                    if new_background is not None:

                        background = new_background

                        previous_mask = None

                    smooth_cursor = None

                elif key == ord(" "):

                    cloak_enabled = not cloak_enabled

                elif key == ord("m"):

                    show_mask = not show_mask

                    if not show_mask:

                        try:

                            cv2.destroyWindow(

                                "Cloth Detection Mask"

                            )

                        except cv2.error:

                            pass

                elif key in (ord("+"), ord("=")):

                    sensitivity = min(

                        100, sensitivity + 5

                    )

                elif key in (ord("-"), ord("_")):

                    sensitivity = max(

                        0, sensitivity - 5

                    )

    except Exception as error:

        print("An error occurred:", error)

        raise

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print("Project closed.")

if __name__ == "__main__":

    main()
