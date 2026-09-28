#!/bin/sh
set -eu
cd "$(dirname "$0")"
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models
curl -L --fail --silent --show-error 'https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' -o models/face_detection_yunet_2023mar.onnx
curl -L --fail --silent --show-error 'https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' -o models/face_recognition_sface_2021dec.onnx
printf '%s\n' \
  '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  models/face_detection_yunet_2023mar.onnx' \
  '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  models/face_recognition_sface_2021dec.onnx' | shasum -a 256 -c
