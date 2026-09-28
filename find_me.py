#!/usr/bin/env python3
"""Find photos containing the enrolled face, entirely on this computer."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


register_heif_opener()


ROOT = Path(__file__).resolve().parent
DETECTOR = ROOT / "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER = ROOT / "models/face_recognition_sface_2021dec.onnx"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def load_photo(path: Path, max_side: int = 3000) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class Matcher:
    def __init__(self) -> None:
        if not DETECTOR.is_file() or not RECOGNIZER.is_file():
            raise RuntimeError("Missing OpenCV models; see README.md")
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR), "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER), "")

    def faces(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(image)
        if detections is None:
            return []
        found = []
        for face in detections:
            crop = self.recognizer.alignCrop(image, face)
            vector = self.recognizer.feature(crop).reshape(-1).astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm:
                found.append((face, vector / norm))
        return found


def photos(folder: Path, excluded: Path | None = None) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
                  and (excluded is None or p != excluded and excluded not in p.parents))


def render_face(image: np.ndarray, detection: np.ndarray, size: int = 220) -> Image.Image:
    x, y, width, height = detection[:4]
    pad = max(width, height) * 0.35
    left, top = max(0, int(x - pad)), max(0, int(y - pad))
    right, bottom = min(image.shape[1], int(x + width + pad)), min(image.shape[0], int(y + height + pad))
    crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    thumb = Image.fromarray(crop)
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    return thumb


def inspect_references(args: argparse.Namespace, matcher: Matcher) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in photos(args.input):
        try:
            image = load_photo(path)
            found = matcher.faces(image)
            for index, (face, _) in enumerate(sorted(found, key=lambda item: item[0][0])):
                name = f"{path.stem}-{index}.jpg"
                render_face(image, face).save(args.output / name, quality=90)
                entries.append((path.name, index, name))
            print(f"{path.name}: {len(found)} faces")
        except Exception as exc:
            print(f"{path.name}: ERROR {exc}", file=sys.stderr)
    body = "\n".join(f"<figure><img src='{html.escape(name)}'><figcaption>{html.escape(source)} — face {index}</figcaption></figure>"
                     for source, index, name in entries)
    (args.output / "index.html").write_text("<meta charset='utf-8'><style>body{font:16px system-ui;background:#f5f4ef;display:flex;flex-wrap:wrap}figure{background:white;padding:12px;margin:8px}img{height:180px;max-width:240px;object-fit:contain}</style>" + body)
    print(f"Face contact sheet: {args.output / 'index.html'}")
    return 0


def scan(args: argparse.Namespace, matcher: Matcher) -> int:
    if args.gallery.is_file():
        with np.load(args.gallery) as data:
            matrix = data["vectors"]
        print(f"Using {len(matrix)} Randy faces from {args.gallery}")
    else:
        references = []
        for path in photos(args.references):
            try:
                found = matcher.faces(load_photo(path, 1600))
                if len(found) != 1:
                    raise ValueError(f"expected one face, found {len(found)}")
                references.append(found[0][1])
            except Exception as exc:
                print(f"Reference {path.name}: {exc}", file=sys.stderr)
        if len(references) < 2:
            raise RuntimeError("Build gallery.npz with enroll_photos.py or add two single-face reference images")
        matrix = np.stack(references)
    paths = photos(args.input, args.output)
    if not paths:
        raise RuntimeError(f"No supported photos found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    thumbs = args.output / "thumbnails"
    thumbs.mkdir(exist_ok=True)
    rows = []
    for number, path in enumerate(paths, 1):
        try:
            image = load_photo(path)
            found = matcher.faces(image)
            scored = sorted(((float(np.mean(np.sort(matrix @ vector)[-min(5, len(matrix)):])), face) for face, vector in found),
                            key=lambda item: item[0], reverse=True)
            score = scored[0][0] if scored else None
            status = "match" if score is not None and score >= args.threshold else (
                "no face" if score is None else "review" if score >= args.review_threshold else "unlikely")
            thumb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            thumb.thumbnail((500, 360), Image.Resampling.LANCZOS)
            for similarity, face in scored:
                x, y, w, h = face[:4]
                sx, sy = thumb.width / image.shape[1], thumb.height / image.shape[0]
                cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                from PIL import ImageDraw
                ImageDraw.Draw(thumb).rectangle((x*sx, y*sy, (x+w)*sx, (y+h)*sy),
                                                outline="#188038" if similarity >= args.threshold else "#d18b00", width=3)
            thumb_name = f"{number:05d}.jpg"
            thumb.save(thumbs / thumb_name, quality=85)
            rows.append({"path": str(path), "status": status, "score": "" if score is None else f"{score:.4f}",
                         "faces": len(found), "thumbnail": f"thumbnails/{thumb_name}"})
        except Exception as exc:
            rows.append({"path": str(path), "status": "error", "score": "", "faces": 0,
                         "thumbnail": "", "error": str(exc)})
        print(f"{number}/{len(paths)} {rows[-1]['status']:9} {rows[-1]['score']:>6} {path.name}", flush=True)
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "status", "score", "faces", "thumbnail", "error"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2))
    cards = []
    for row in sorted(rows, key=lambda item: (item["status"] != "match", -(float(item["score"] or 0)))):
        title = html.escape(Path(row["path"]).name)
        thumbnail = html.escape(row["thumbnail"])
        link = Path(row["path"]).resolve().as_uri()
        cards.append(f"<article class='{row['status'].replace(' ', '-')}'><a href='{html.escape(link)}'>"
                     f"<img src='{thumbnail}' alt='Preview of {title}'><b>{title}</b></a>"
                     f"<small>{html.escape(row['status'])} · score {row['score'] or '—'} · {row['faces']} faces</small></article>")
    counts = {status: sum(row["status"] == status for row in rows) for status in ["match", "review", "unlikely", "no face", "error"]}
    style = "body{font:16px system-ui;margin:30px;background:#f5f4ef;color:#17211b}header{position:sticky;top:0;background:#f5f4ef;padding:8px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}article{background:white;padding:12px;border-radius:8px}article.match{border:3px solid #188038}article.review{border:3px solid #d18b00}img{width:100%;height:220px;object-fit:contain;background:#eee}b,small{display:block;overflow-wrap:anywhere}small{margin-top:6px}"
    (args.output / "index.html").write_text("<meta charset='utf-8'><title>Find me in photos</title><style>" + style + "</style>"
        + f"<header><h1>Find me in photos</h1><p>{len(rows)} photos · "
        + " · ".join(f"{name}: {count}" for name, count in counts.items())
        + "</p><p>Green = suggested match. Orange = check manually. Scores are similarity, not probabilities.</p></header><main>"
        + "\n".join(cards) + "</main>")
    print(f"Results: {args.output / 'index.html'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Make a contact sheet of detected faces for reference selection")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--output", type=Path, default=ROOT / "reference_candidates")
    search = commands.add_parser("scan", help="Scan a folder for Randy")
    search.add_argument("input", type=Path)
    search.add_argument("--gallery", type=Path, default=ROOT / "gallery.npz")
    search.add_argument("--references", type=Path, default=ROOT / "reference_faces")
    search.add_argument("--output", type=Path, default=ROOT / "results")
    search.add_argument("--threshold", type=float, default=0.58)
    search.add_argument("--review-threshold", type=float, default=0.45)
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"No such folder: {args.input}")
    if args.command == "scan" and not args.gallery.is_file() and not args.references.is_dir():
        parser.error("No gallery.npz or reference_faces folder; run enroll_photos.py first")
    matcher = Matcher()
    return inspect_references(args, matcher) if args.command == "inspect" else scan(args, matcher)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
