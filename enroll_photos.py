#!/usr/bin/env python3
"""Build a local face gallery from every face tagged Randy in Apple Photos."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from find_me import Matcher


register_heif_opener()


def tagged_faces(library: Path, person: str):
    database = library / "database/Photos.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    people = connection.execute("SELECT Z_PK, ZFULLNAME, ZFACECOUNT FROM ZPERSON WHERE ZFULLNAME = ? ORDER BY ZFACECOUNT DESC", (person,)).fetchall()
    if not people:
        raise RuntimeError(f"No Photos person named {person!r}")
    person_id = people[0]["Z_PK"]
    query = """SELECT a.Z_PK asset_id, a.ZDIRECTORY directory, a.ZFILENAME filename,
                      f.ZCENTERX center_x, f.ZCENTERY center_y, f.ZSIZE face_size
               FROM ZDETECTEDFACE f JOIN ZASSET a ON a.Z_PK = f.ZASSETFORFACE
               WHERE f.ZPERSONFORFACE = ? AND a.ZKIND = 0 AND a.ZTRASHEDSTATE = 0
                 AND a.ZHIDDEN = 0 AND a.ZVISIBILITYSTATE = 0
               ORDER BY a.ZDATECREATED DESC"""
    records = connection.execute(query, (person_id,)).fetchall()
    connection.close()
    return person_id, records


def tagged_crop(path: Path, x: float, y_bottom: float, size: float) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        # Photos stores upright normalized coordinates with the origin at bottom left.
        cx, cy = x * width, (1 - y_bottom) * height
        radius = max(0.06 * min(width, height), 2.2 * size * max(width, height))
        box = (max(0, int(cx-radius)), max(0, int(cy-radius)),
               min(width, int(cx+radius)), min(height, int(cy+radius)))
        image = image.crop(box)
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path.home() / "Pictures/Photos Library.photoslibrary")
    parser.add_argument("--person", default="Randy R")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "gallery.npz")
    parser.add_argument("--limit", type=int, default=0, help="For a quick diagnostic run")
    args = parser.parse_args()
    if not args.library.is_dir():
        parser.error(f"No such Photos library: {args.library}")
    person_id, records = tagged_faces(args.library, args.person)
    if args.limit:
        records = records[:args.limit]
    print(f"Photos person {args.person!r} ({person_id}): {len(records)} tagged photo faces", flush=True)
    matcher = Matcher()
    vectors, sources, failures = [], [], []
    source_counts = {"original": 0, "preview": 0, "thumbnail": 0}
    for number, row in enumerate(records, 1):
        path = args.library / "originals" / row["directory"] / row["filename"]
        try:
            if not path.is_file():
                preview = args.library / "resources/derivatives" / row["directory"] / (Path(row["filename"]).stem + "_1_105_c.jpeg")
                thumbnail = args.library / "resources/derivatives/masters" / row["directory"] / (Path(row["filename"]).stem + "_4_5005_c.jpeg")
                path = preview if preview.is_file() else thumbnail
                source_type = "preview" if preview.is_file() else "thumbnail"
            else:
                source_type = "original"
            if not path.is_file():
                raise FileNotFoundError("no local original or Photos preview")
            crop = tagged_crop(path, row["center_x"], row["center_y"], row["face_size"])
            faces = matcher.faces(crop)
            if not faces:
                raise ValueError("no face detected near Photos tag")
            # The Photos face location, rather than other people in the image, determines enrollment.
            midpoint = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
            face, vector = min(faces, key=lambda entry: np.linalg.norm(
                np.array([entry[0][0] + entry[0][2]/2, entry[0][1] + entry[0][3]/2]) - midpoint))
            if np.linalg.norm(np.array([face[0]+face[2]/2, face[1]+face[3]/2]) - midpoint) > .35 * max(crop.shape[:2]):
                raise ValueError("detected face too far from Photos tag")
            vectors.append(vector)
            sources.append(f"{row['asset_id']}:{row['filename']}")
            source_counts[source_type] += 1
        except Exception as error:
            failures.append({"asset": row["asset_id"], "file": row["filename"], "error": str(error)})
        if number % 100 == 0 or number == len(records):
            print(f"{number}/{len(records)} processed · {len(vectors)} faces · {len(failures)} skipped", flush=True)
    if not vectors:
        raise RuntimeError("No Photos tagged faces could be enrolled")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vectors=np.stack(vectors), sources=np.array(sources))
    report = {"person": args.person, "person_id": person_id, "tagged_faces": len(records),
              "enrolled_faces": len(vectors), "skipped": len(failures), "source_counts": source_counts,
              "failures": failures}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"Gallery: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
