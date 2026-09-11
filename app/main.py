import time
import cv2
from ultralytics import YOLO

def run_live_detection(source=0):
    """
    Runs real-time object detection on a live video stream.
    Source: 0 for local webcam, or a string URL for mobile camera stream.
    """
    print("[INFO] Loading YOLOv8 Nano model...")
    model = YOLO("models/yolov8n.pt")

    # Initialize video capture stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print(f"[INFO] Stream started on source {source}. Press 'q' in the preview window to exit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from source.")
            break

        # Run inference (conf=0.4 filters out weak detections)
        results = model(frame, conf=0.4, verbose=False)

        # Parse detections for the Context-Aware Engine
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # Get bounding box coordinates [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                
                # Calculate bounding box center (x_center, y_center)
                x_center = (xyxy[0] + xyxy[2]) / 2.0
                y_center = (xyxy[1] + xyxy[3]) / 2.0

        # Render annotations on the video frame
        annotated_frame = results[0].plot()

        # Calculate and overlay real-time FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display window
        cv2.imshow("CAVE - Real-Time Vision Feed", annotated_frame)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # source=0 defaults to your integrated PC webcam.
    run_live_detection(source=0)