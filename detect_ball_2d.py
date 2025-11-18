import cv2
import numpy as np

INPUT_VIDEO_PATH = "./rgb.avi"
OUTPUT_VIDEO_PATH = "output_red_ball_terrace.avi"

# ------------------- HELPERS -------------------

def create_video_capture(path):
    """
    Open a video source (file path or camera index) and return a cv2.VideoCapture.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {path}")
    return cap


def preprocess_frame(frame, target_width=960):
    """
    Optionally resize the frame to a given width while keeping aspect ratio.
    """
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w != target_width:
        scale = target_width / float(w)
        frame = cv2.resize(frame, (target_width, int(h * scale)))
    return frame


def get_red_mask(frame):
    """
    Create a binary mask for a red ball in HSV color space.
    Combines two red ranges (0-10 and 170-180 degrees in Hue).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower reds
    lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    # upper reds
    lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def find_red_ball(
    mask,
    frame_shape,
    min_area=120,        # slightly higher to ignore tiny blobs
    max_area_ratio=0.2,
    min_circularity=0.6
):
    """
    Find the most likely red ball contour:
    - only in the lower part of the frame (ball on the floor/terrace)
    - filtering by area and circularity
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    h, w = frame_shape[:2]
    max_area = h * w * max_area_ratio
    horizon_y = int(h * 0.35)  # ignore everything too high in the image

    best_circle = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), int(radius)

        # the ball must be below the "horizon" line
        if y < horizon_y:
            continue

        # simple score: circularity * area (tunable)
        score = circularity * area
        if score > best_score:
            best_score = score
            best_circle = (x, y, radius)

    return best_circle


def draw_ball(frame, circle):
    """
    Draw the detected ball as a red circle with a red center dot.
    No coordinates on screen, just a visual marker.
    """
    if circle is None:
        return frame

    x, y, r = circle
    cv2.circle(frame, (x, y), r, (0, 0, 255), 3)     # outer circle
    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)    # center

    # Optional label
    cv2.putText(
        frame,
        "Ball",
        (x - 20, y - r - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return frame

# ------------------- MAIN -------------------

def main():
    cap = create_video_capture(INPUT_VIDEO_PATH)

    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch frame.")
            break

        frame = preprocess_frame(frame, target_width=960)

        # Initialize video writer after we know final frame size
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

        # 1. red mask
        red_mask = get_red_mask(frame)

        # 2. find the ball
        circle = find_red_ball(red_mask, frame.shape)

        # 3. draw result
        out_frame = frame.copy()
        out_frame = draw_ball(out_frame, circle)

        cv2.imshow("Red Ball Detection - Terrace", out_frame)
        out.write(out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
