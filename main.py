import cv2
from ultralytics import YOLO
from utils.draw import draw_bbox
from trackers.deepsort_tracker import create_tracker

# Danh sách class COCO
COCO = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

def main():
    # 1) Load model YOLOv5u
    model = YOLO("models/yolov5su.pt")  # Sẽ tự tải nếu chưa có

    # 2) Khởi tạo tracker
    tracker = create_tracker()

    # 3) Mở video
    cap = cv2.VideoCapture("data/demo1.mp4")  # Đổi thành 0 để dùng webcam

    if not cap.isOpened():
        print("[ERROR] Không mở được video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4) Detect bằng YOLO
        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
        res = results[0]

        # 5) Chuyển đổi bbox cho DeepSORT
        dets = []
        for b in res.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            dets.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])

        # 6) Tracking
        tracks = tracker.update_tracks(dets, frame=frame)

        # 7) Vẽ kết quả
        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            l, t0, w, h = t.to_ltwh()
            x1, y1, x2, y2 = int(l), int(t0), int(l + w), int(t0 + h)
            cls_id = getattr(t, "det_class", None)
            label = f"ID {track_id}"
            if cls_id is not None and 0 <= cls_id < len(COCO):
                label = f"{COCO[cls_id]} | {label}"
            draw_bbox(frame, x1, y1, x2, y2, label)

        cv2.imshow("YOLOv5u + Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
