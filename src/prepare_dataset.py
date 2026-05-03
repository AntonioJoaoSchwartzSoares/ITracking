import argparse
import random
import shutil
from pathlib import Path

import cv2


def extract_frames(video_path: Path, raw_images_dir: Path, every_n: int, max_frames: int) -> int:
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir video: {video_path}")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n == 0:
            out_name = f"{video_path.stem}_f{frame_idx:06d}.jpg"
            out_path = raw_images_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def split_train_val(raw_images_dir: Path, train_dir: Path, val_dir: Path, val_ratio: float, seed: int) -> tuple[int, int]:
    imgs = sorted(raw_images_dir.glob("*.jpg"))
    if not imgs:
        return 0, 0

    random.seed(seed)
    random.shuffle(imgs)

    val_count = int(len(imgs) * val_ratio)
    val_set = set(imgs[:val_count])

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for p in imgs:
        target = val_dir if p in val_set else train_dir
        shutil.copy2(p, target / p.name)

    return len(imgs) - val_count, val_count


def ensure_label_files(images_dir: Path, labels_dir: Path) -> int:
    labels_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for img_path in images_dir.glob("*.jpg"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")
            created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara dataset YOLO para classe kart.")
    parser.add_argument("--video", required=True, help="Video de entrada")
    parser.add_argument("--dataset-root", default="/app/datasets/kart", help="Raiz do dataset")
    parser.add_argument("--every-n", type=int, default=8, help="Salva 1 frame a cada N")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proporcao de validacao")
    parser.add_argument("--seed", type=int, default=42, help="Seed para split")
    parser.add_argument("--skip-split", action="store_true", help="Extrai apenas para raw_images")
    args = parser.parse_args()

    video_path = Path(args.video)
    dataset_root = Path(args.dataset_root)

    raw_images_dir = dataset_root / "raw_images"
    images_train_dir = dataset_root / "images" / "train"
    images_val_dir = dataset_root / "images" / "val"
    labels_train_dir = dataset_root / "labels" / "train"
    labels_val_dir = dataset_root / "labels" / "val"

    if not video_path.exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")

    saved = extract_frames(video_path, raw_images_dir, args.every_n, args.max_frames)
    print(f"Frames extraidos em raw_images: {saved}")

    if args.skip_split:
        print("Split pulado (--skip-split).")
        return

    train_count, val_count = split_train_val(raw_images_dir, images_train_dir, images_val_dir, args.val_ratio, args.seed)
    print(f"Split concluido: train={train_count} val={val_count}")

    created_train = ensure_label_files(images_train_dir, labels_train_dir)
    created_val = ensure_label_files(images_val_dir, labels_val_dir)
    print(f"Arquivos de label vazios criados: train={created_train} val={created_val}")
    print("Agora anote os .txt com formato YOLO: <class x_center y_center width height> normalizados.")


if __name__ == "__main__":
    main()

