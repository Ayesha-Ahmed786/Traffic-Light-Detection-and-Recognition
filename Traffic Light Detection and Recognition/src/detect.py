import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils import get_signal_color
import config

# Load YOLO
model = YOLO(config.MODEL_PATH)

# find class id
def find_class_id(names, target="traffic light"):
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == target.lower():
                return int(k)
    else:
        try:
            return int(list(names).index(target))
        except ValueError:
            return None
    return None

traffic_id = find_class_id(model.names, "traffic light")
device = '0' if torch.cuda.is_available() else 'cpu'

def process_image(img_path, save_path="output.jpg"):
    results = model.predict(
        source=img_path,
        conf=config.CONFIDENCE,
        imgsz=config.IMG_SIZE,
        classes=[traffic_id],
        device=device,
        verbose=False
    )
    r = results[0]
    img = cv2.imread(img_path)

    if hasattr(r, "boxes") and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1,y1,x2,y2 = xyxy[i].astype(int)
            conf = confs[i]

            crop = img[y1:y2, x1:x2]
            color = get_signal_color(crop)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{color} {conf:.2f}",
                        (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)

    cv2.imwrite(save_path, img)
    print(f"✅ Image processed and saved to {save_path}")

def process_video(video_path, save_path="output.avi"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w,h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(
            source=frame,
            conf=config.CONFIDENCE,
            imgsz=config.IMG_SIZE,
            classes=[traffic_id],
            device=device,
            verbose=False
        )
        r = results[0]

        if hasattr(r, "boxes") and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1,y1,x2,y2 = xyxy[i].astype(int)
                conf = confs[i]

                crop = frame[y1:y2, x1:x2]
                color = get_signal_color(crop)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{color} {conf:.2f}",
                            (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames. Saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Light Detection with YOLOv8")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save results")

    args = parser.parse_args()

    if args.image:
        process_image(args.image, args.output)
    elif args.video:
        process_video(args.video, args.output)
    else:
        print("⚠️ Please provide either --image or --video")
