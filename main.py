import argparse
import cv2
import time
from ultralytics import YOLO
from utils.draw import draw_bbox
from trackers.deepsort_tracker import create_tracker
from motion.motion_analysis import MotionAnalyzer  # THÊM: phân tích chuyển động

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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="", help="Đường dẫn video; bỏ trống = webcam")
    ap.add_argument("--weights", type=str, default="models/yolov5su.pt", help="YOLOv5u weights")
    ap.add_argument("--imgsz", type=int, default=640, help="Kích thước input")
    ap.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
    ap.add_argument("--device", type=str, default=None, help="cuda hoặc cpu (auto nếu để trống)")
    ap.add_argument("--classes", type=str, default="person", help="Các class cần theo dõi, ví dụ: person,bicycle,car")
    return ap.parse_args()

def safe_name(cls_id: int) -> str:
    return COCO[cls_id] if 0 <= cls_id < len(COCO) else str(cls_id)

def main():
    args = parse_args()

    # 1) Load YOLOv5u
    model = YOLO(args.weights)

    # 2) Khởi tạo DeepSORT
    tracker = create_tracker()

    # 3) Khởi tạo MotionAnalyzer
    analyzer = MotionAnalyzer(fps=30)

    # 4) Mở nguồn video
    cap = cv2.VideoCapture("data/demo2.mp4") if args.video == "" else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Không mở được nguồn video.")
        return

    # 5) Danh sách class cần theo dõi
    wanted = set([s.strip() for s in args.classes.split(",") if s.strip()])

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 6) Dự đoán
        results = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )
        res = results[0]

        # 7) Chuẩn bị input cho DeepSORT
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                cls_id = int(b.cls[0])
                name = safe_name(cls_id)
                if name not in wanted:
                    continue
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(float, b.xyxy[0])
                dets.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])

        # 8) Tracking
        tracks = tracker.update_tracks(dets, frame=frame)
        timestamp = time.time()

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            l, t0, w, h = t.to_ltwh()
            x1, y1, x2, y2 = int(l), int(t0), int(l + w), int(t0 + h)

            cls_id = getattr(t, "det_class", None)
            label = f"ID {tid}"
            cname = "unknown"

            if cls_id is not None:
                cname = safe_name(int(cls_id))
                label = f"{cname} | ID {tid}"

            # 9) Phân tích chuyển động
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            analyzer.update(tid, timestamp, center_x, center_y)

            motion = analyzer.get_features_and_label(
                tid,
                cls_name=cname,
                cfg={
                    "thresholds": {
                        "pxps_walk": 28,
                        "pxps_run": 85,
                        "by_class": {
                            "person": {
                                "pxps_walk": 28,
                                "pxps_run": 85
                            }
                        }
                    }
                }
            )

            if motion:
                label += f" | {motion['label']} ({motion['speed']:.1f}px/s)"

            draw_bbox(frame, x1, y1, x2, y2, label)

        # 10) Hiển thị kết quả
        cv2.namedWindow("YOLOv5u + DeepSORT (Tracking)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv5u + DeepSORT (Tracking)", 960, 540)
        cv2.imshow("YOLOv5u + DeepSORT (Tracking)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
