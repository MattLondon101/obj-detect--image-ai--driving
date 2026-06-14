# Video Object Detection
Code and instructions for computer vision object detection in live video.

## **ImageAI Video Object Detection**
ImageAI is excellent Python software for Computer Vison object detection in still images and video.
This repository focuses on the latter.

Video files in .avi format work best. A small sample video and a large database are accessed 
at [the Cambridge-driving Labeled Video Database (CamVid)](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

This first example exhibits, first the sample video pre-object detection processing, then the video processed with the TinyYOLOv3 model and ImageAI.


Input Video (41 seconds) : Video from driving automobile
Click link to view video on YouTube, then click your browser's "Back" button to return here.

[![](http://i3.ytimg.com/vi/7m-O_Zfpcfs/hqdefault.jpg)](https://www.youtube.com/embed/7m-O_Zfpcfs)


Output Video (41 seconds) : Video from driving automobile processed with the TinyYOLOv3 model and ImageAI
Click link to view video on YouTube, then click your browser's "Back" button to return here.

[![](http://i3.ytimg.com/vi/lHv4tpx3QhU/hqdefault.jpg)](https://youtube.com/embed/lHv4tpx3QhU)


Follow link to Google Colab notebook for instructions to get started: [ImageAI1.ipynb](https://colab.research.google.com/drive/1cn5bzIk71YOeRdhbfJNRD7IgZx1xfgJu)

## External data setup

Keep large images, videos, datasets, and model weights outside Git. This repo is
set up so those files are declared in `ImageAI-master/data-assets.json` and
downloaded locally when needed:

```bash
UV_CACHE_DIR=.uv-cache uv venv .venv
UV_CACHE_DIR=.uv-cache uv run scripts/fetch_assets.py
```

To add a new external asset, upload it to an authoritative source such as a
release asset, object storage bucket, dataset registry, or signed internal URL.
Then add an entry to `ImageAI-master/data-assets.json`:

```json
{
  "path": "ImageAI-master/data-videos/traffic.mp4",
  "url": "https://your-storage.example/traffic.mp4",
  "sha256": "optional checksum",
  "extract": "optional/destination/for/archives"
}
```

The `.gitignore` excludes the local media and model paths so newly downloaded
files are not accidentally committed.

This repository currently has media files already tracked in Git. Ignore rules
do not remove files that are already tracked, so do this once after confirming
the manifest points at the correct online sources:

```bash
git rm -r --cached ImageAI-master/data-images \
  ImageAI-master/data-videos \
  ImageAI-master/test-images \
  ImageAI-master/test/data-images \
  ImageAI-master/test/data-videos \
  ImageAI-master/test/data-datasets
git add .gitignore ImageAI-master/data-assets.json scripts/fetch_assets.py README.md
git commit -m "Move media assets to external downloads"
```

## Run video inference

This repository vendors ImageAI 2.1.5, which uses TensorFlow 1-era Keras APIs.
It is not compatible with modern `keras` 3.x. The commands below must use a
legacy Python/Keras/TensorFlow environment:

```bash
UV_CACHE_DIR=.uv-cache uv venv --python 3.7 .venv-imageai
UV_CACHE_DIR=.uv-cache uv pip install -p .venv-imageai setuptools wheel
UV_CACHE_DIR=.uv-cache uv pip install -p .venv-imageai --no-build-isolation -e ./ImageAI-master \
  "tensorflow==1.14.0" \
  "keras==2.2.4" \
  "numpy<1.19" \
  "h5py<3" \
  "opencv-python<4.3" \
  "scipy<1.5" \
  pillow matplotlib
```

Download a pretrained ImageAI object detection model from the official ImageAI
release assets:

https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0

TinyYOLOv3 is smaller and faster, so it is the recommended first model:

```bash
curl -L -o ImageAI-master/yolo-tiny.h5 \
  https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5
```

YOLOv3 is larger and usually more accurate:

```bash
curl -L -o ImageAI-master/yolo.h5 \
  https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
```

Run the video inference helper against `ImageAI-master/data-videos`:

```bash
.venv-imageai/bin/python scripts/run_video_inference.py \
  --model ImageAI-master/yolo-tiny.h5 \
  --model-type tiny-yolo \
  --input ImageAI-master/data-videos
```

Outputs are written to `ImageAI-master/output-videos/*.avi`. Use `--input` with
a single file such as `ImageAI-master/data-videos/traffic-mini.mp4` to process
one video. If you downloaded `yolo.h5`, pass `--model ImageAI-master/yolo.h5`
and `--model-type yolo`.
