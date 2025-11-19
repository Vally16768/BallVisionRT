import cv2


def create_video_capture(path: str) -> cv2.VideoCapture:
    """Open a video source and return a cv2.VideoCapture."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {path}")
    return cap


def preprocess_frame(frame, target_width):
    """Optionally resize the frame to a given width while keeping aspect ratio."""
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / float(w)
    new_size = (target_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def create_video_writer(path: str, fps: float, frame_size):
    """Create a VideoWriter with given FPS and frame size."""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {path}")
    return writer
