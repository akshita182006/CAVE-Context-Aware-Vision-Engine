import cv2
import numpy as np

from PIL import Image
from transformers import pipeline

print("[INFO] Loading Depth Anything V2...")

depth_model = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("[INFO] Depth model loaded.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Could not read frame.")
        break

    # OpenCV = BGR, model input = RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    # Estimate depth
    result = depth_model(image)

    depth_image = np.array(result["depth"])

    # Normalize for visualization
    depth_normalized = cv2.normalize(
        depth_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Create visible depth map
    depth_colored = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_INFERNO
    )

    cv2.imshow("CAVE - Camera", frame)
    cv2.imshow("CAVE - Depth Map", depth_colored)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 