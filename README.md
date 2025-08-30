# 3D Depth Estimation with Depth Anything V2

This project provides a web-based service for generating high-quality depth maps from images and videos using the state-of-the-art **Depth Anything V2 (DAV2)** model. The generated depth maps are used to create a compelling, interactive 3D parallax effect in a real-time web viewer built with Three.js.

## Features

- **State-of-the-Art Depth Estimation**: Leverages the Depth Anything V2 model for highly accurate and detailed monocular depth prediction.
- **Image and Video Support**: Process both static images (JPG, PNG, WEBP) and videos (MP4, AVI, WEBM).
- **Interactive 3D Viewer**: A performant Three.js frontend displaces a mesh in 3D space based on the depth map, responding to user mouse/touch input.
- **GPU Acceleration**: Automatically utilizes an available NVIDIA CUDA GPU for significantly faster inference, with a seamless fallback to CPU if needed or in case of Out-of-Memory errors.
- **Advanced Video Stabilization**: Implements a flicker-reduction pipeline for video processing, using global normalization and motion-aware temporal smoothing (via optical flow) to ensure stable and consistent depth output.
- **High-Precision Depth Output**: For images, depth maps can be saved as 16-bit packed PNGs, preserving fine detail and eliminating the banding artifacts common in 8-bit formats.
- **Configurable Performance**: Key parameters like model size for video, inference resolution, and tiled processing can be configured via environment variables.

## Core Functionality
- **Web Service:** The script creates a web application using **FastAPI** to serve a user interface and process media files.
- **Deep Learning Model:** It leverages the **Depth Anything V2 (DAV2)** model from Hugging Face for state-of-the-art monocular depth estimation.
- **Dual-Model Strategy:** It uses the high-quality **Large (L)** version of DAV2 for still images and the faster **Small (S)** version for video frames to balance quality and performance (this is configurable).
- **GPU Acceleration:** It automatically detects and utilizes a CUDA-enabled GPU for faster inference, with a built-in fallback to CPU if the GPU runs out of memory (OOM).

#### Image Processing Pipeline
- **User Interface:** Provides a simple drag-and-drop web page for users to upload images (JPG, PNG, WEBP).
- **Depth Map Generation:** For each uploaded image, it generates a high-precision depth map.
- **High-Precision Depth Storage:** It features an option to pack the depth map into a pseudo 16-bit format within a PNG's Red and Green channels (`SAVE_DEPTH_RG16`). This offers 65,536 depth levels instead of the standard 256, significantly reducing banding artifacts.
- **Normal Map Generation:** It also computes a normal map from the depth data, which is used in the 3D viewer to enhance lighting and surface details.
- **Tiled Inference:** Includes an optional tiled processing mode (`USE_TILED`) to handle very high-resolution images without running out of GPU memory, by processing the image in overlapping chunks.

#### Video Processing Pipeline
- **Video Upload:** Accepts common video formats (MP4, AVI, WEBM).
- **Transcoding:** Uses **ffmpeg** to automatically transcode uploaded videos into a standardized, web-compatible MP4 (H.264) format, preserving audio.
- **Frame-by-Frame Processing:** It processes the video frame by frame to generate a corresponding depth video.
- **Flicker Reduction (Temporal Stability):** This is a key feature. To ensure the depth video is stable and doesn't "flicker," it employs several advanced techniques:
    - **Global Normalization:** It can perform an initial calibration pass on a subset of video frames to establish a consistent minimum and maximum depth range for the entire video.
    - **Motion-Aware Smoothing:** It uses optical flow to warp the depth map from the previous frame to align with the current frame. It then blends the new depth map with this warped-previous map, resulting in a temporally smooth and stable effect.
- **Dithering for Quality:** It applies a stable Bayer dithering pattern to the final 8-bit depth video, which effectively breaks up color/gradient banding without introducing temporal noise (flickering).

#### 3D Interactive Viewer (Frontend)
- **Real-time 3D Effect:** It uses **Three.js** to create an interactive 3D viewer directly in the browser.
- **Displacement Mapping:** The 3D effect is not fake parallax. It uses a highly tessellated plane mesh where each vertex is displaced in 3D space according to the depth map, creating a true 3D relief effect.
- **Interactive Parallax:** The viewpoint shifts based on the user's mouse or touch position, creating an immersive "look-around" feeling. The sensitivity is biased towards the center for a more natural interaction.
- **Post-Processing Pipeline:** The viewer includes a sophisticated post-processing pipeline for maximum visual quality:
    - **SMAA (Antialiasing):** For smooth, crisp edges without jagged lines.
    - **Dynamic Depth of Field (DOF):** The background and foreground blur subtly as the user interacts, enhancing the sense of depth.
    - **Sharpening & Dithering:** A subtle sharpening filter is applied, and a final dithering step prevents banding in gradients.
    - **Adaptive Softening:** Automatically applies a gentle blur when the image or video is upscaled to fill the screen, reducing pixelation.

## The Model: Depth Anything V2

The core of this project is **Depth Anything V2 (DAV2)**, a powerful foundation model for monocular depth estimation. It excels at understanding scene geometry from a single 2D image and produces robust depth maps that generalize well to a wide variety of contexts.

- **Automatic Download**: The required models (both Large and Small variants) are downloaded and cached automatically from the Hugging Face Hub on the first run. You do not need to download them manually.
- **Dual-Model Strategy**:
  - **Images**: By default, the high-quality `Depth-Anything-V2-Large` model is used for maximum detail.
  - **Videos**: For performance, the faster `Depth-Anything-V2-Small` model is used for frame-by-frame processing. This can be overridden to use the Large model for higher-quality video output at the cost of speed.

## Setup and Installation

### Prerequisites

- **Python 3.8+** and Pip
- **Git**
- **ffmpeg**: Required for video processing (transcoding and frame extraction). Make sure it is installed and accessible in your system's PATH. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).
- (Optional) **NVIDIA GPU** with CUDA drivers installed for hardware acceleration.

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/ssbagpcm/image-to-3D-v2.git
    cd image-to- 3D-v2
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The project's dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For optimal performance with a GPU, consider installing the CUDA-enabled version of PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).*

4.  **Hugging Face Authentication** (Optional)
    To prevent potential rate-limiting when downloading models from the Hugging Face Hub, it is recommended to authenticate. Create an access token on [huggingface.co](https://huggingface.co/settings/tokens) and set it as an environment variable.

    Create a `.env` file in the project root and add your token:
    ```
    HUGGINGFACE_HUB_TOKEN="hf_YourTokenGoesHere"
    ```
    The application will automatically load this variable.

## Configuration

The application can be configured using environment variables. You can set them in your shell or add them to the `.env` file.

| Variable                  | Default | Description                                                               |
| ------------------------- | ------- | ------------------------------------------------------------------------- |
| `DAV2_MAX_EDGE`           | `2048`  | Maximum resolution edge for image processing.                             |
| `DAV2_VIDEO_MAX_EDGE`     | `1024`  | Maximum resolution edge for video frame processing.                       |
| `USE_LARGE_FOR_VIDEO`     | `0`     | Set to `1` to use the higher-quality Large model for videos (slower).       |
| `DAV2_TILED`              | `0`     | Set to `1` to enable tiled inference for very high-resolution images.       |
| `DEPTH_PNG_RG16`          | `1`     | Set to `1` to save image depth maps in high-precision 16-bit format.        |
| `VIDEO_CALIBRATE`         | `1`     | Set to `1` to enable the anti-flicker calibration pre-pass for videos.    |
| `VIDEO_CRF`               | `14`    | The CRF (Constant Rate Factor) for ffmpeg video transcoding. Lower is higher quality. |


## Running the Application

Once the setup is complete, run the web server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
- `main`: Refers to the `main.py` file.
- `app`: Refers to the `FastAPI` instance named `app` inside `main.py`.

The server will be available at `http://127.0.0.1:8000`.

## Usage

1.  **Open the Web Interface**: Navigate to `http://127.0.0.1:8000` in your web browser.
2.  **Upload a File**: Drag and drop an image or video file onto the upload area, or use the "Choose file" button.
3.  **Processing**: The application will process the file. Images are typically fast, while videos will take longer depending on their duration and resolution.
4.  **View the Result**: Once processing is complete, you will be automatically redirected to the interactive 3D viewer.

### Demo Files
To get started quickly, this repository includes a `demos/` directory containing sample images that you can use to test the application immediately.

## Technology Stack

- **Backend**: FastAPI, PyTorch, Transformers, OpenCV
- **Frontend**: Three.js, HTML5, JavaScript
- **ML Model**: Depth Anything V2 (from Hugging Face)
