import argparse
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
    model = YOLO(args.weights)  # sẽ tự tải nếu chưa có

    # 2) Khởi tạo DeepSORT
    tracker = create_tracker()

    # 3) Nguồn video
    cap = cv2.VideoCapture("data/demo1.mp4") if args.video == "" else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Không mở được nguồn video.")
        return

    # 4) Danh sách class cần theo dõi (set để tra cứu nhanh)
    wanted = set([s.strip() for s in args.classes.split(",") if s.strip()])

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 5) Detect
        results = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )
        res = results[0]

        # 6) Chuẩn bị detections cho DeepSORT: [[x,y,w,h], conf, class_id]
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                cls_id = int(b.cls[0])
                name = safe_name(cls_id)
                if name not in wanted:
                    continue  # LỌC NGAY TỪ ĐÂY (ví dụ: chỉ 'person')
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(float, b.xyxy[0])
                dets.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])

        # 7) Tracking
        tracks = tracker.update_tracks(dets, frame=frame)

        # 8) Vẽ kết quả
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            l, t0, w, h = t.to_ltwh()
            x1, y1, x2, y2 = int(l), int(t0), int(l + w), int(t0 + h)

            # Lấy class name từ detection gần nhất (nếu có)
            cls_id = getattr(t, "det_class", None)
            label = f"ID {tid}"
            if cls_id is not None:
                cname = safe_name(int(cls_id))
                label = f"{cname} | ID {tid}"

            draw_bbox(frame, x1, y1, x2, y2, label)

        cv2.imshow("YOLOv5u + DeepSORT (Tracking)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
