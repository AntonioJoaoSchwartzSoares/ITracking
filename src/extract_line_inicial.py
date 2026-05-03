"""
Extracao de tracado no estilo do script inicial do projeto:
YOLO + track (persist), ponto no centro da base da bbox, lista de (x, y),
imagem final desenhada sobre o primeiro frame.

Diferenca em relacao ao exemplo em loop: com varios objetos no frame,
o exemplo original acrescentava um ponto por caixa (varios pontos no mesmo
instante). Aqui gravamos um unico ponto por frame — a deteccao com maior
confianca (como no extract_line.py da raiz), mais estavel para uma linha 1D.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def ponto_base_bbox_xywh(box_xywh):
    x, y, w, h = box_xywh
    return int(x), int(y + h / 2.0)


def main():
    parser = argparse.ArgumentParser(
        description="Traçado estilo script inicial: YOLO track + ponto na base da bbox + imagem no 1º frame."
    )
    parser.add_argument("--video", required=True, help="Vídeo de entrada")
    parser.add_argument("--out", required=True, help="Pasta de saída")
    parser.add_argument("--model", default="yolo11n.pt", help="Peso YOLO (ex.: yolo11n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima")
    parser.add_argument(
        "--class-id",
        type=int,
        default=2,
        help="Classe COCO (2=car). Use -1 para não filtrar classes.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Mostrar janela OpenCV (precisa de DISPLAY; em Docker costuma ser desnecessário).",
    )
    parser.add_argument(
        "--write-overlay",
        action="store_true",
        help="Gravar overlay.mp4 com a linha verde por cima do vídeo.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, background_frame = cap.read()
    if not ret:
        raise RuntimeError("Não foi possível ler o primeiro frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(args.model)

    classes_kw = {}
    if args.class_id is not None and args.class_id >= 0:
        classes_kw["classes"] = [args.class_id]

    tracado_pontos = []  # (frame_idx, x, y, conf)
    writer = None
    if args.write_overlay:
        overlay_path = out_dir / "overlay.mp4"
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

    print("Processando vídeo... (q na janela interrompe se --preview)")

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            verbose=False,
            **classes_kw,
        )

        boxes_obj = results[0].boxes if results else None
        if boxes_obj is not None and len(boxes_obj) > 0:
            xywh = boxes_obj.xywh.cpu().numpy()
            confs = boxes_obj.conf.cpu().numpy()
            j = int(np.argmax(confs))
            x, y = ponto_base_bbox_xywh(xywh[j])
            c = float(confs[j])
            tracado_pontos.append((frame_idx, x, y, c))

        vis = frame.copy()
        pts_xy = [(p[1], p[2]) for p in tracado_pontos]
        for i in range(1, len(pts_xy)):
            cv2.line(vis, pts_xy[i - 1], pts_xy[i], (0, 255, 0), 2, cv2.LINE_AA)

        if args.preview:
            cv2.putText(
                vis,
                f"frame {frame_idx + 1}/{nframes} pts={len(tracado_pontos)}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Acompanhamento de Traçado", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    # Mapa final: copia do primeiro frame + linha azul (BGR) como no exemplo (255, 0, 0)
    print("Gerando mapa de traçado final...")
    mapa = background_frame.copy()
    pts_xy = [(p[1], p[2]) for p in tracado_pontos]
    for i in range(1, len(pts_xy)):
        cv2.line(mapa, pts_xy[i - 1], pts_xy[i], (255, 0, 0), 3, cv2.LINE_AA)

    out_png = out_dir / "resultado_tracado.png"
    cv2.imwrite(str(out_png), mapa)

    csv_path = out_dir / "trajetoria.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "x", "y", "conf"])
        wcsv.writerows(tracado_pontos)

    print(f"OK: {out_png}")
    print(f"OK: {csv_path}")
    if writer is not None:
        print(f"OK: {out_dir / 'overlay.mp4'}")
    print(f"Resumo: pontos={len(tracado_pontos)} frames={frame_idx}")


if __name__ == "__main__":
    main()
