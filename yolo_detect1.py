import os
import cv2
import time
import numpy as np
from ultralytics import YOLO


# -----------------------------
# USER CONFIGURATION
# -----------------------------

# Path to your trained YOLO model
MODEL_PATH = "weights/best.pt"   # 🔹 Change this if needed

# Minimum confidence threshold
CONF_THRESH = 0.5

# Desired output resolution (None = keep default webcam resolution)
RESOLUTION = "1200x720"  # or None

# Whether to record output
RECORD = False


# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

print(f"✅ Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH, task="detect")
labels = model.names
print(f"✅ Model loaded successfully with {len(labels)} classes.")


# -----------------------------
# INITIALIZE WEBCAM
# -----------------------------
print("🎥 Opening built-in webcam...")
cap = cv2.VideoCapture(0)  # 0 = built-in / default camera

if not cap.isOpened():
    raise RuntimeError("❌ Could not access webcam. Make sure it's connected or not in use by another app.")

# Set resolution if defined
if RESOLUTION:
    resW, resH = map(int, RESOLUTION.split("x"))
    cap.set(3, resW)
    cap.set(4, resH)
else:
    resW = int(cap.get(3))
    resH = int(cap.get(4))

# Setup recorder if enabled
if RECORD:
    recorder = cv2.VideoWriter(
        "webcam_output.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (resW, resH)
    )
    print("📹 Recording enabled: webcam_output.avi")

# Define bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168), (169,162,241),
               (98,118,150), (172,176,184)]

# -----------------------------
# DETECTION LOOP
# -----------------------------
fps_buffer = []
fps_avg_len = 30
avg_fps = 0

print("🚀 Press 'Q' to quit | 'P' to save snapshot")

while True:
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("❌ Unable to read from webcam. Exiting...")
        break

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # Draw detections
    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        conf = det.conf.item()
        cls_id = int(det.cls.item())
        class_name = labels[cls_id]

        if conf >= CONF_THRESH:
            color = bbox_colors[cls_id % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{class_name}: {conf*100:.1f}%"
            cv2.putText(frame, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            object_count += 1

    # FPS calculation
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)

    # Display info
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show result
    cv2.imshow("YOLO - Built-in Webcam", frame)

    if RECORD:
        recorder.write(frame)

    # Key bindings
    key = cv2.waitKey(5)
    if key in [ord("q"), ord("Q")]:
        print("👋 Exiting...")
        break
    elif key in [ord("p"), ord("P")]:
        cv2.imwrite("snapshot.png", frame)
        print("💾 Snapshot saved as snapshot.png")

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
if RECORD:
    recorder.release()
cv2.destroyAllWindows()
print(f"✅ Average FPS: {avg_fps:.2f}")

