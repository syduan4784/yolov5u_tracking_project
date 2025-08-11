import cv2

def draw_bbox(frame, x1, y1, x2, y2, label, color=(36, 255, 12), thickness=2):
    """Vẽ bounding box và label lên frame"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
