import cv2
import numpy as np
from pathlib import Path

out = Path("/app/data/input/volta1.mp4")
out.parent.mkdir(parents=True, exist_ok=True)

w, h, fps = 640, 360, 30
vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

for i in range(120):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, ((50 + i * 4) % w, 180), 20, (255, 255, 255), -1)
    vw.write(img)

vw.release()
print("ok", out)
