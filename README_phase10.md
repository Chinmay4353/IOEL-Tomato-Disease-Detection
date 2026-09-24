# Phase 10: Raspberry Pi Deployment Guide

This phase prepares the project for deployment on a Raspberry Pi.

## 1. Raspberry Pi OS setup

1. Install Raspberry Pi OS Lite or Desktop.
2. Connect to Wi-Fi or Ethernet.
3. Update the system:

```bash
sudo apt update
sudo apt upgrade -y
```

## 2. Python installation

Check Python version:

```bash
python3 --version
```

If needed:

```bash
sudo apt install python3 python3-pip python3-venv -y
```

## 3. Virtual environment

```bash
cd ~/tomato-detector
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Required packages

```bash
pip install tensorflow opencv-python pillow numpy matplotlib requests
```

For a leaner inference setup, keep only the packages actually required for edge inference.

## 5. Camera configuration

For Raspberry Pi Camera Module:

```bash
sudo raspi-config
```

Enable:
- Interface Options
- Legacy Camera or Camera

Verify camera:

```bash
libcamera-hello --list-cameras
```

## 6. Copy model

Place the converted model and required files into:

```bash
~/tomato-detector/model/
```

## 7. Copy inference code

Copy:
- src/inference/camera_app.py
- src/inference/tflite_predict.py
- src/data/preprocessing.py
- src/utils/config.py

## 8. Run application

Example:

```bash
python3 src/inference/camera_app.py --camera-index 0
```

## 9. Troubleshooting

- Camera not detected: check cable and enable camera interface
- TensorFlow import errors: install the correct Python version and dependencies
- Low performance: reduce frame rate and use TFLite model
- Permission errors: ensure the user has access to the project directory

## 10. Final Raspberry Pi layout

```text
tomato-detector/
├── model/
│   ├── model.tflite
│   └── labels.txt
├── src/
│   ├── camera_app.py
│   ├── tflite_predict.py
│   ├── preprocessing.py
│   └── config.py
├── captured_images/
├── logs/
├── requirements.txt
├── README.md
└── .venv/
```

## Objective

This guide prepares the project for the edge deployment step, where the Raspberry Pi captures an image and runs the TFLite model locally.
