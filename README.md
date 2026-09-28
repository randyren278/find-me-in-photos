# Find me in photos

This local tool finds photos likely to contain Randy. It detects every face in each image, compares each face with the Apple Photos **Randy R** face tags, and writes an HTML review page and CSV. It never uploads photos or face data.

## Set up

Requires macOS, `uv`, and access to the local Photos library. From this directory:

```sh
./setup.sh
.venv/bin/python enroll_photos.py
```

Enrollment reads the Photos database and existing local originals or previews. It does not change the Photos library. The private `gallery.npz` file contains face embeddings; keep it private. `gallery.json` records how many tagged faces were usable and why any were skipped. Re-run enrollment after Photos has added new Randy pictures.

## Scan a folder

```sh
./find-me /path/to/photos --output /path/to/results
```

Open `index.html` in the output folder. Green photos are suggested matches, orange photos need review, and the CSV includes every file scanned. Scores are similarities, **not probabilities**. A face hidden by sunglasses, turned away, or very small can be missed; group photos can receive a high score for another person. Check the results before sharing or deleting anything. Originals are never moved or changed.

Supported inputs: JPEG, PNG, WebP, BMP, TIFF, and HEIC/HEIF. Folders are scanned recursively. Videos are not scanned. You can adjust `--threshold` (default `0.58`) and `--review-threshold` (default `0.45`).

The gallery is made from Photos' internal database and preview files, whose layout may change with macOS updates. If the Photos layout changes, use `find_me.py inspect /path/to/reference-photos` to make a face contact sheet and place at least two **single-face crops of Randy** in `reference_faces/`; scanning uses those when `gallery.npz` is absent.

Face detection and matching use [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Model checksums are verified by `setup.sh`.
