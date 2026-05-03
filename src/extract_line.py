import argparse
import csv
from collections import defaultdict
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


def crop_safe(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def hsv_hist_signature(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def hist_similarity(h1, h2):
    if h1 is None or h2 is None:
        return -1.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def save_csv(points, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "conf", "method", "track_id"])
        w.writerows(points)


def save_ids_inicio(ids_rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "x", "y", "conf"])
        w.writerows(ids_rows)


def save_ids_resumo(ids_rows, out_path):
    stats = defaultdict(lambda: {"n": 0, "fmin": 10**9, "fmax": -1, "sx": 0.0, "sy": 0.0, "sconf": 0.0})

    for fr, tid, x, y, conf in ids_rows:
        d = stats[int(tid)]
        d["n"] += 1
        d["fmin"] = min(d["fmin"], int(fr))
        d["fmax"] = max(d["fmax"], int(fr))
        d["sx"] += float(x)
        d["sy"] += float(y)
        d["sconf"] += float(conf)

    rows = []
    for tid, d in stats.items():
        rows.append(
            [
                tid,
                d["n"],
                d["fmin"],
                d["fmax"],
                round(d["sx"] / d["n"], 2),
                round(d["sy"] / d["n"], 2),
                round(d["sconf"] / d["n"], 4),
            ]
        )

    rows.sort(key=lambda r: r[1], reverse=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "samples", "first_frame", "last_frame", "mean_x", "mean_y", "mean_conf"])
        w.writerows(rows)


def auto_choose_target_id(ids_rows, start_point=None, min_travel=120.0, min_motion_span=50.0):
    """
    Escolhe automaticamente o melhor track_id com base em:
    - frequencia de aparicao
    - continuidade temporal (menos gaps)
    - confianca media
    - proximidade do ponto inicial (opcional) como desempate
    """
    if not ids_rows:
        return None

    by_id = defaultdict(list)
    for fr, tid, x, y, conf in ids_rows:
        tid = int(tid)
        if tid < 0:
            continue
        by_id[tid].append((int(fr), float(x), float(y), float(conf)))

    if not by_id:
        return None

    sx, sy = start_point if start_point is not None else (None, None)

    best_id = None
    best_score = -1e18
    for tid, rows in by_id.items():
        rows.sort(key=lambda r: r[0])

        frames = [r[0] for r in rows]
        n = len(rows)
        mean_conf = float(np.mean([r[3] for r in rows]))
        mean_x = float(np.mean([r[1] for r in rows]))
        mean_y = float(np.mean([r[2] for r in rows]))
        first_x, first_y = rows[0][1], rows[0][2]

        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
        gap_penalty = sum(max(0, g - 1) for g in gaps)

        # Penaliza objetos quase parados (ex: carro de apoio estacionado).
        travel = 0.0
        for i in range(1, len(rows)):
            travel += float(np.hypot(rows[i][1] - rows[i - 1][1], rows[i][2] - rows[i - 1][2]))
        motion_span = float(np.hypot(max(r[1] for r in rows) - min(r[1] for r in rows), max(r[2] for r in rows) - min(r[2] for r in rows)))
        static_penalty = 1200.0 if (travel < min_travel and motion_span < min_motion_span and n > 12) else 0.0

        score = 3.0 * n + 50.0 * mean_conf - 2.0 * gap_penalty
        score += 0.35 * travel + 0.6 * motion_span
        score -= static_penalty

        if sx is not None and sy is not None:
            # Distancia ao ponto inicial tem peso alto para evitar selecionar objeto errado.
            mean_dist = np.hypot(mean_x - sx, mean_y - sy)
            first_dist = np.hypot(first_x - sx, first_y - sy)
            score -= 0.25 * mean_dist
            score -= 0.55 * first_dist

        if score > best_score:
            best_score = score
            best_id = tid

    return best_id


def fallback_id_nearest_start(ids_rows, start_point):
    """Se auto_choose_target_id devolver None, escolhe o track_id com a amostra mais proxima de start_point."""
    if not ids_rows or start_point is None:
        return None
    sx, sy = float(start_point[0]), float(start_point[1])
    by_tid = defaultdict(list)
    for _fr, tid, x, y, _conf in ids_rows:
        tid = int(tid)
        if tid < 0:
            continue
        by_tid[tid].append((float(x), float(y)))
    if not by_tid:
        return None
    best_tid = None
    best_d = 1e18
    for tid, pts in by_tid.items():
        dmin = min((px - sx) ** 2 + (py - sy) ** 2 for px, py in pts)
        if dmin < best_d:
            best_d = dmin
            best_tid = tid
    return best_tid


def fallback_id_most_common(ids_rows):
    """Ultimo recurso sem start_point: track_id mais frequente no scan."""
    tids = [int(r[1]) for r in ids_rows if int(r[1]) >= 0]
    if not tids:
        return None
    return max(set(tids), key=tids.count)


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


def interpolate_short_gaps(points, max_gap=6):
    if len(points) < 2:
        return points

    out = [points[0]]
    for i in range(1, len(points)):
        f0, x0, y0, c0, m0, id0 = points[i - 1]
        f1, x1, y1, c1, m1, id1 = points[i]
        gap = f1 - f0

        if 1 < gap <= max_gap + 1:
            for k in range(1, gap):
                t = k / gap
                xi = int((1 - t) * x0 + t * x1)
                yi = int((1 - t) * y0 + t * y1)
                fi = f0 + k
                out.append((fi, xi, yi, min(c0, c1), "interp", id0))
        out.append(points[i])

    out.sort(key=lambda r: r[0])
    return out


def run_yolo_track(
    model,
    frame,
    conf,
    class_id,
    imgsz,
    yolo_min_area,
    yolo_max_area,
    yolo_min_ratio,
    yolo_max_ratio,
    yolo_edge_margin,
    include_rect,
    exclude_rect,
):
    kwargs = {"source": frame, "persist": True, "conf": conf, "verbose": False, "imgsz": imgsz}
    if class_id is not None:
        kwargs["classes"] = [class_id]

    results = model.track(**kwargs)
    if not results or len(results) == 0:
        return []

    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    xywh = boxes_obj.xywh.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    ids = boxes_obj.id.cpu().numpy().astype(int) if boxes_obj.id is not None else None

    dets = []
    h, w = frame.shape[:2]
    for i in range(len(xywh)):
        x, y, bw, bh = xywh[i]
        area = float(bw * bh)
        if area < yolo_min_area or area > yolo_max_area:
            continue
        ratio = float(bw / max(bh, 1e-6))
        if ratio < yolo_min_ratio or ratio > yolo_max_ratio:
            continue
        px = int(x)
        py = int(y + bh / 2.0)
        if (
            px < yolo_edge_margin
            or px > (w - yolo_edge_margin)
            or py < yolo_edge_margin
            or py > (h - yolo_edge_margin)
        ):
            continue
        if include_rect is not None:
            rx1, ry1, rx2, ry2 = include_rect
            if not (rx1 <= px <= rx2 and ry1 <= py <= ry2):
                continue
        if exclude_rect is not None:
            ex1, ey1, ex2, ey2 = exclude_rect
            if ex1 <= px <= ex2 and ey1 <= py <= ey2:
                continue
        x1 = int(x - bw / 2.0)
        y1 = int(y - bh / 2.0)
        x2 = int(x + bw / 2.0)
        y2 = int(y + bh / 2.0)
        tid = int(ids[i]) if ids is not None else -1

        dets.append(
            {"px": px, "py": py, "conf": float(confs[i]), "bbox": (x1, y1, x2, y2), "method": "yolo", "track_id": tid}
        )
    return dets


def detect_with_motion_cv(frame, bg_sub, prev_point=None, roi_top_ratio=0.10, min_area=30, max_area=900):
    h, w = frame.shape[:2]
    y0 = int(h * roi_top_ratio)
    y0 = max(0, min(y0, h - 1))
    roi = frame[y0:, :]

    fg = bg_sub.apply(roi)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e9
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        ratio = bw / max(bh, 1)
        if ratio < 0.35 or ratio > 3.5:
            continue

        xg, yg = x, y + y0
        px = int(xg + bw / 2.0)
        py = int(yg + bh / 2.0)

        score = float(area)
        if prev_point is not None:
            d = np.hypot(px - prev_point[0], py - prev_point[1])
            score -= d * 2.0

        if score > best_score:
            best_score = score
            best = (xg, yg, xg + bw, yg + bh, px, py)

    if best is None:
        return None

    x1, y1, x2, y2, px, py = best
    return {"px": px, "py": py, "conf": 1.0, "bbox": (x1, y1, x2, y2), "method": "cv-motion", "track_id": -1}


def pick_target_id(dets, start_point):
    if not dets:
        return None
    sx, sy = start_point
    best_id = None
    best_d = 1e18
    for d in dets:
        if d["track_id"] < 0:
            continue
        dd = (d["px"] - sx) ** 2 + (d["py"] - sy) ** 2
        if dd < best_d:
            best_d = dd
            best_id = d["track_id"]
    return best_id


def main():
    parser = argparse.ArgumentParser(description="Racing line extraction with hard ID lock + ids_inicio.csv")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamanho de inferencia YOLO (ex: 960, 1280, 1600)")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--class-id", type=int, default=2, help="2=car, -1=no class filter")
    parser.add_argument("--yolo-min-area", type=float, default=20.0, help="Area minima da bbox YOLO em pixels")
    parser.add_argument("--yolo-max-area", type=float, default=1800.0, help="Area maxima da bbox YOLO em pixels")
    parser.add_argument("--yolo-min-ratio", type=float, default=0.45, help="Razao minima largura/altura da bbox YOLO")
    parser.add_argument("--yolo-max-ratio", type=float, default=2.8, help="Razao maxima largura/altura da bbox YOLO")
    parser.add_argument("--yolo-edge-margin", type=int, default=45, help="Ignora deteccoes perto da borda da imagem")
    parser.add_argument("--auto-min-travel", type=float, default=120.0, help="Movimento minimo para auto-pick de ID")
    parser.add_argument("--auto-min-span", type=float, default=50.0, help="Span minimo (caixa de movimento) para auto-pick")
    parser.add_argument("--force-target-after-scan", action=argparse.BooleanOptionalAction, default=True, help="Exige target_id escolhido apos ids-scan-frames")
    parser.add_argument("--include-rect", type=str, default=None, help="Regiao permitida x1,y1,x2,y2 (pixels)")
    parser.add_argument(
        "--exclude-rect",
        type=str,
        default=None,
        help="Regiao a ignorar x1,y1,x2,y2 (ex.: box do carro de apoio). Ponto-base da bbox dentro deste retangulo e descartado.",
    )
    parser.add_argument(
        "--prioritize-yolo-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quando houver target_id, nao troca para outro ID e evita fallback CV.",
    )

    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument("--start-x", type=int, default=None)
    parser.add_argument("--start-y", type=int, default=None)
    parser.add_argument("--init-frames", type=int, default=30)

    parser.add_argument("--hard-lock-id", action="store_true")
    parser.add_argument("--max-gap-frames", type=int, default=60)
    parser.add_argument("--max-jump", type=float, default=30.0)
    parser.add_argument("--lock-radius", type=float, default=60.0)

    parser.add_argument("--lost-max-frames", type=int, default=20)
    parser.add_argument("--reacquire-radius", type=float, default=50.0)
    parser.add_argument("--min-hist-sim", type=float, default=0.58)

    parser.add_argument("--no-fallback-cv", action="store_true")
    parser.add_argument("--roi-top-ratio", type=float, default=0.10)
    parser.add_argument("--cv-min-area", type=float, default=30.0)
    parser.add_argument("--cv-max-area", type=float, default=900.0)

    parser.add_argument("--interp-gap", type=int, default=6)
    parser.add_argument("--ids-scan-frames", type=int, default=120, help="Quantos frames iniciais exportar em ids_inicio.csv")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(args.model)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    class_id = None if args.class_id < 0 else args.class_id
    include_rect = None
    if args.include_rect:
        parts = [int(p.strip()) for p in args.include_rect.split(",")]
        if len(parts) != 4:
            raise ValueError("--include-rect deve ser x1,y1,x2,y2")
        x1, y1, x2, y2 = parts
        include_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    exclude_rect = None
    if args.exclude_rect:
        parts = [int(p.strip()) for p in args.exclude_rect.split(",")]
        if len(parts) != 4:
            raise ValueError("--exclude-rect deve ser x1,y1,x2,y2")
        x1, y1, x2, y2 = parts
        exclude_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    overlay_path = out_dir / "overlay.mp4"
    heatmap_path = out_dir / "heatmap.png"
    csv_path = out_dir / "trajetoria.csv"
    ids_inicio_path = out_dir / "ids_inicio.csv"
    ids_resumo_path = out_dir / "ids_resumo.csv"

    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    target_id = args.target_id
    auto_select_enabled = args.auto_select and args.start_x is not None and args.start_y is not None
    start_point = (args.start_x, args.start_y) if auto_select_enabled else None

    points = []
    ids_rows = []
    frame_idx = 0
    auto_pick_done = False

    yolo_hits = 0
    cv_hits = 0
    rejected_jump = 0
    blocked_switch = 0

    lost_frames = 0
    last_pos = None
    last_vel = (0.0, 0.0)
    target_hist = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = run_yolo_track(
            model,
            frame,
            conf=args.conf,
            class_id=class_id,
            imgsz=args.imgsz,
            yolo_min_area=args.yolo_min_area,
            yolo_max_area=args.yolo_max_area,
            yolo_min_ratio=args.yolo_min_ratio,
            yolo_max_ratio=args.yolo_max_ratio,
            yolo_edge_margin=args.yolo_edge_margin,
            include_rect=include_rect,
            exclude_rect=exclude_rect,
        )

        # Export IDs in first N frames
        if frame_idx < args.ids_scan_frames:
            for d in dets:
                if d["track_id"] >= 0:
                    ids_rows.append((frame_idx, d["track_id"], d["px"], d["py"], d["conf"]))

        # Quando nao vier --target-id, tenta escolher automaticamente
        # apos acumular frames iniciais de scan.
        if (
            target_id is None
            and not auto_pick_done
            and frame_idx >= max(1, args.ids_scan_frames - 1)
        ):
            sp = (args.start_x, args.start_y) if (args.start_x is not None and args.start_y is not None) else None
            target_id = auto_choose_target_id(
                ids_rows,
                start_point=sp,
                min_travel=args.auto_min_travel,
                min_motion_span=args.auto_min_span,
            )
            if target_id is None and ids_rows:
                if sp is not None:
                    target_id = fallback_id_nearest_start(ids_rows, sp)
                    if target_id is not None:
                        print(f"[AUTO] fallback (mais proximo de start): target_id={target_id}")
                if target_id is None:
                    target_id = fallback_id_most_common(ids_rows)
                    if target_id is not None:
                        print(f"[AUTO] fallback (mais frequente no scan): target_id={target_id}")
            auto_pick_done = True
            print(f"[AUTO] target_id escolhido: {target_id}")

        # Se configurado, nao escolhe candidato enquanto nao houver target_id definido.
        if args.force_target_after_scan and frame_idx >= args.ids_scan_frames and target_id is None:
            vis = frame.copy()
            cv2.putText(
                vis,
                "Aguardando target_id (auto-pick falhou)",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(vis)
            frame_idx += 1
            continue

        chosen = None

        if target_id is None and auto_select_enabled and frame_idx < args.init_frames:
            tid = pick_target_id(dets, start_point)
            if tid is not None:
                target_id = tid

        if args.hard_lock_id and target_id is not None:
            same_id = [d for d in dets if d["track_id"] == target_id]
            if same_id:
                chosen = max(same_id, key=lambda d: d["conf"])
                lost_frames = 0
            else:
                chosen = None
                lost_frames += 1
        else:
            pred = None
            if last_pos is not None:
                pred = (last_pos[0] + last_vel[0], last_pos[1] + last_vel[1])

            if target_id is not None:
                same_id = [d for d in dets if d["track_id"] == target_id]
                if same_id:
                    chosen = max(same_id, key=lambda d: d["conf"])
                    lost_frames = 0
                else:
                    lost_frames += 1
            else:
                if dets:
                    chosen = max(dets, key=lambda d: d["conf"])

            if (
                chosen is None
                and target_id is not None
                and pred is not None
                and lost_frames <= args.lost_max_frames
                and dets
                and (not args.prioritize_yolo_id)
            ):
                candidates = []
                for d in dets:
                    dist = np.hypot(d["px"] - pred[0], d["py"] - pred[1])
                    if dist > args.reacquire_radius:
                        continue

                    x1, y1, x2, y2 = d["bbox"]
                    patch = crop_safe(frame, x1, y1, x2, y2)
                    if patch is None:
                        continue

                    hsig = hsv_hist_signature(patch)
                    sim = hist_similarity(target_hist, hsig) if target_hist is not None else 1.0
                    if sim < args.min_hist_sim:
                        continue

                    score = d["conf"] + 0.5 * sim - 0.01 * dist
                    candidates.append((score, d, hsig))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    _, chosen, best_hist = candidates[0]
                    if chosen["track_id"] >= 0:
                        target_id = chosen["track_id"]
                    target_hist = best_hist
                    lost_frames = 0

        if chosen is not None and last_pos is not None:
            dlock = np.hypot(chosen["px"] - last_pos[0], chosen["py"] - last_pos[1])
            if dlock > args.lock_radius:
                chosen = None
                blocked_switch += 1

        if args.prioritize_yolo_id and target_id is not None:
            allow_cv = False
        elif args.hard_lock_id and target_id is not None:
            allow_cv = False
        else:
            allow_cv = (target_id is None) or (lost_frames > args.lost_max_frames)

        if chosen is None and (not args.no_fallback_cv) and allow_cv:
            chosen = detect_with_motion_cv(
                frame=frame,
                bg_sub=bg_sub,
                prev_point=last_pos,
                roi_top_ratio=args.roi_top_ratio,
                min_area=args.cv_min_area,
                max_area=args.cv_max_area,
            )

        vis = frame.copy()

        if chosen is not None:
            px, py = chosen["px"], chosen["py"]
            x1, y1, x2, y2 = chosen["bbox"]
            conf = float(chosen["conf"])
            method = chosen["method"]
            tid = int(chosen["track_id"])

            accept = True
            if last_pos is not None:
                dj = np.hypot(px - last_pos[0], py - last_pos[1])
                if dj > args.max_jump:
                    accept = False
                    rejected_jump += 1

            if accept:
                points.append((frame_idx, px, py, conf, method, tid))
                if method == "yolo":
                    yolo_hits += 1
                else:
                    cv_hits += 1

                if last_pos is not None:
                    vx = px - last_pos[0]
                    vy = py - last_pos[1]
                    last_vel = (0.7 * last_vel[0] + 0.3 * vx, 0.7 * last_vel[1] + 0.3 * vy)
                last_pos = (px, py)

                if method == "yolo":
                    patch = crop_safe(frame, x1, y1, x2, y2)
                    if patch is not None:
                        hsig = hsv_hist_signature(patch)
                        target_hist = hsig if target_hist is None else (0.8 * target_hist + 0.2 * hsig)

                color = (0, 200, 255) if method == "yolo" else (255, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.circle(vis, (px, py), 4, color, -1)
                label = f"id={tid}" if tid >= 0 else "id=cv"
                cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        else:
            if args.hard_lock_id and target_id is not None:
                lost_frames += 1
                if lost_frames > args.max_gap_frames:
                    pass

        if len(points) > 1:
            line_pts = [(p[1], p[2]) for p in points]
            line_pts = ema_smooth(line_pts, alpha=0.25)
            draw_polyline(vis, line_pts, (0, 255, 0), 2)

        cv2.putText(vis, f"frame {frame_idx + 1}/{total_frames}", (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"target_id={target_id} hard={int(args.hard_lock_id)} lost={lost_frames} pts={len(points)} yolo={yolo_hits} cv={cv_hits}",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    save_ids_inicio(ids_rows, ids_inicio_path)
    save_ids_resumo(ids_rows, ids_resumo_path)

    if len(points) == 0:
        raise RuntimeError("No points detected. Verifique ids_resumo.csv e tente --target-id ou ajuste --conf/--class-id.")

    points = sorted(points, key=lambda r: r[0])
    points = interpolate_short_gaps(points, max_gap=args.interp_gap)

    save_csv(points, csv_path)
    points_xy = [(p[1], p[2]) for p in points]
    make_heatmap(first_frame, points_xy, heatmap_path)

    print(f"OK: {overlay_path}")
    print(f"OK: {heatmap_path}")
    print(f"OK: {csv_path}")
    print(f"OK: {ids_inicio_path}")
    print(f"OK: {ids_resumo_path}")
    print(f"Summary: total_points={len(points)} target_id={target_id} yolo_hits={yolo_hits} cv_hits={cv_hits}")


if __name__ == "__main__":
    main()