# YOLOv10 Workbench

A self-hosted FastAPI workbench for Ultralytics detection training and inference.

## Features

- Responsive light and dark themes with a persisted theme preference.
- A focused Train and Predict workflow with all compatible expert parameters grouped by purpose.
- Pretrained model downloads with checksum verification, local `.pt` paths, and collision-safe uploads.
- Image, video, path, and URL prediction sources.
- One managed Ultralytics process at a time, with live logs, training metrics, output galleries, and browser-ready video.
- A local Runs archive that reads existing artifacts from `runs/train` and `runs/predict` without migration.

## Installation

Python 3.10 or later is required.

```bash
conda create -n yolov10-workbench python=3.10
conda activate yolov10-workbench
pip install -r requirements.txt
pip install -e .
```

## Start

```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Set `YOLOV10_WEBUI_PORT` or `YOLOV10_WEBUI_HOST` to override the default binding.

The application is self-hosted, has no authentication layer, and runs one training or inference command at a time to protect the shared model process.
