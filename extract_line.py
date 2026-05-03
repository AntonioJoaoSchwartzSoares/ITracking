import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def ema_smooth(points, alpha=0.25):
    if len(points) < 2:
        return points
    smoothed = [points[0]]
    for p in points[1:]:
        prev = smoothed[-1]
        x = int(alpha * p[0] + (1.0 - alpha) * prev[0])
        y = int(alpha * p[1] + (1.0 - alpha) * prev[1])
        smoothed.append((x, y))
    return smoothed


def draw_polyline(img, points, color, thickness=2):
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        cv2.line(img, points[i - 1], points[i], color, thickness, cv2.LINE_AA)


def detect_with_yolo(model, frame, conf, class_id=None):
    # NOTE: source=frame works in Ultralytics for per-frame inference/tracking
    kwargs = {"source": frame, "persist": True, "conf": conf, "verbose": False}
    if class_id is not None:
        kwargs["classes"] = [class_id]

    results = model.track(**kwargs)
    if not results or len(results) == 0:
        return None

    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return None

    boxes = boxes_obj.xywh.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    x, y, bw, bh = boxes[best_i]
    score = float(confs[best_i])

    # Centro da base da bbox (aprox. contato no asfalto)
    px = int(x)
    py = int(y + bh / 2.0)

    # bbox em xyxy para visualizacao
    x1 = int(x - bw / 2.0)
    y1 = int(y - bh / 2.0)
    x2 = int(x + bw / 2.0)
    y2 = int(y + bh / 2.0)

    return {"px": px, "py": py, "conf": score, "bbox": (x1, y1, x2, y2), "method": "yolo"}


def detect_with_classic_cv(frame, thresh=200, min_area=50):
    # Fallback para video sintetico ou casos em que YOLO nao detecta
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(c)

    # Centro da base da bbox
    px = int(x + w / 2.0)
    py = int(y + h)

    return {"px": px, "py": py, "conf": 1.0, "bbox": (x, y, x + w, y + h), "method": "cv"}


def save_csv(points, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "conf", "method"])
        writer.writerows(points)


def make_heatmap(first_frame, points_xy, out_path):
    h, w = first_frame.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for x, y in points_xy:
        if 0 <= x < w and 0 <= y < h:
            heat[y, x] += 1.0

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=15, sigmaY=15)
    if heat.max() > 0:
        heat_norm = (heat / heat.max() * 255.0).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)

    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(first_frame, 0.6, heat_color, 0.4, 0)

    smooth_final = ema_smooth(points_xy, alpha=0.20)
    draw_polyline(blended, smooth_final, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), blended)


def main():
    parser = argparse.ArgumentParser(description="Extracao de tracado (racing line) a partir de video.")
    parser.add_argument("--video", required=True, help="Video de entrada")
    parser.add_argument("--out", required=True, help="Diretorio de saida")
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo YOLO")
    parser.add_argument("--conf", type=float, default=0.15, help="Confianca minima do YOLO")
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Classe COCO (ex: 2=car). Omitir ou -1 = sem filtro (recomendado se yolo_hits=0 com classe 2).",
    )
    parser.add_argument("--max-jump", type=float, default=90.0, help="Salto maximo aceito entre frames")
    parser.add_argument("--cv-thresh", type=int, default=200, help="Threshold fallback CV")
    parser.add_argument("--cv-min-area", type=float, default=50.0, help="Area minima fallback CV")
    parser.add_argument("--no-fallback-cv", action="store_true", help="Desabilita fallback CV")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Nao foi possivel ler o primeiro frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(args.model)
    coco_class = None if args.class_id is None or args.class_id < 0 else args.class_id

    overlay_path = out_dir / "overlay.mp4"
    heatmap_path = out_dir / "heatmap.png"
    csv_path = out_dir / "trajetoria.csv"

    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    points = []  # (frame_idx, x, y, conf, method)
    frame_idx = 0
    yolo_hits = 0
    cv_hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = detect_with_yolo(model, frame, conf=args.conf, class_id=coco_class)

        if det is None and not args.no_fallback_cv:
            det = detect_with_classic_cv(frame, thresh=args.cv_thresh, min_area=args.cv_min_area)

        vis = frame.copy()

        if det is not None:
            px, py = det["px"], det["py"]
            x1, y1, x2, y2 = det["bbox"]
            score = float(det["conf"])
            method = det["method"]

            # Filtro de salto irreal
            accept = True
            if points:
                dx = px - points[-1][1]
                dy = py - points[-1][2]
                if np.hypot(dx, dy) > args.max_jump:
                    accept = False

            if accept:
                points.append((frame_idx, px, py, score, method))
                if method == "yolo":
                    yolo_hits += 1
                else:
                    cv_hits += 1

            color = (0, 200, 255) if method == "yolo" else (255, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (px, py), 4, color, -1)

        if len(points) > 1:
            line_pts = [(p[1], p[2]) for p in points]
            line_pts = ema_smooth(line_pts, alpha=0.25)
            draw_polyline(vis, line_pts, (0, 255, 0), 2)

        cv2.putText(
            vis,
            f"frame {frame_idx + 1}/{total_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"points={len(points)} yolo={yolo_hits} cv={cv_hits}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    if len(points) == 0:
        raise RuntimeError(
            "Nenhum ponto detectado. Dicas: reduzir --conf, usar --class-id None, "
            "habilitar fallback CV (padrao) ou testar video real."
        )

    save_csv(points, csv_path)
    points_xy = [(p[1], p[2]) for p in points]
    make_heatmap(first_frame, points_xy, heatmap_path)

    print(f"OK: {overlay_path}")
    print(f"OK: {heatmap_path}")
    print(f"OK: {csv_path}")
    print(f"Resumo: total_points={len(points)} yolo_hits={yolo_hits} cv_hits={cv_hits}")
    if yolo_hits == 0 and cv_hits > 0:
        print(
            "AVISO: Nenhum ponto veio do YOLO (só fallback CV por contorno). "
            "O heatmap/CSV podem não representar o kart. Tente sem --class-id ou com --class-id -1; "
            "se o kart não for classe COCO 'car' (2), o filtro 2 remove todas as deteções."
        )


if __name__ == "__main__":
    main()