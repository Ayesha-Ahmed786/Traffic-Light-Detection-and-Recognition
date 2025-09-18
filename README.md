# 🚦 Traffic Light Detection and Signal Recognition

This project detects **traffic lights** in images and videos using a pretrained **YOLOv8 model** and then recognizes the **signal color (Red, Yellow, Green)** using **OpenCV color analysis**.  
It can process images, videos, and live webcam streams.

---

## 📖 How It Works

1. **Detection (YOLOv8)**
   - The YOLOv8 model detects objects in a frame.
   - We filter detections to only keep the **`traffic light`** class.
   - Bounding boxes are drawn around detected traffic lights.

2. **Color Recognition (OpenCV)**
   - Once a traffic light is detected, the region inside the bounding box is cropped.
   - The cropped area is converted to **HSV color space** (better for color segmentation).
   - Average color values are analyzed:
     - **Red** → High hue values around 0° or 180°
     - **Yellow** → Hue values around 30°
     - **Green** → Hue values around 60–90°
   - A label (`Red`, `Yellow`, or `Green`) is added on top of the detection box.

3. **Output**
   - For **images**: The result is displayed with bounding boxes + labels.
   - For **videos**: Each frame is processed and saved into a new video file with detections.

---

## 📂 Project Structure
  
- **src/** → Source code folder containing main logic  
  - **config.py** → Stores configuration (file paths, detection thresholds, constants)  
  - **detect.py** → Main script to run YOLOv8 detection and traffic light color recognition  
  - **utils.py** → Helper functions for tasks like color detection (HSV masking), drawing boxes, etc.  
- **requirements.txt** → Python dependencies needed to run the project  

## ⚙️ Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/your-username/traffic-light-detection.git
cd traffic-light-detection
pip install -r requirements.txt
```

## Usage

First install the requirements via:

```bash
pip install -r requirements.txt
```

**Run on Image**

```bash
python src/detect.py --source path/to/image.jpg
```

**Run on Video**

```bash
python src/detect.py --source path/to/video.mp4
```

## Output

- Traffic lights are detected by YOLOv8.

- Signal colors are recognized (Red, Yellow, Green) using HSV thresholds.

- Results are displayed for images and saved as a new video for videos.


## Results

**Image**

<img width="186" height="203" alt="traffic light result" src="https://github.com/user-attachments/assets/1b9e5c58-10ab-49f4-a58e-dda9a2a310dc" />

**Video**



