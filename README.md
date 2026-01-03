# Short Video Caption Creator

A specialized tool for automating video captioning, featuring high-performance transcription, speaker diarization, and dynamic overlay controls.

This project repurposes the OpenAI Whisper architecture (via **WhisperX**) to provide precise, word-level timestamp alignment and speaker identification, significantly outperforming standard transcription methods in speed and accuracy.

## Features

- **High-Performance Transcription**: Utilizes `whisperx` with CTranslate2 for up to 5x faster processing than standard Whisper.
- **Accurate Alignment**: Phoneme-based alignment ensures captions appear exactly when spoken.
- **Speaker Diarization**: automatically detects and labels different speakers (supports adding unique colors per speaker).
- **GPU Acceleration**: Built-in support for CUDA and NVIDIA NVENC hardware encoding for faster rendering.
- **Flexible Composition**:
  - **Overlay Mode**: Add captions to a background video (e.g., gameplay footage).
  - **Caption-Only Mode**: Caption the source video directly without any background overlay.
- **Customizable Output**:
  - Control caption positioning (`top`, `center`, `bottom`).
  - Font styling, sizing, and specific speaker colors configurability.

## Installation

### Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) (Installed and added to system PATH)
- **Optional (for GPU support)**: NVIDIA Driver + CUDA Toolkit 11.8 or higher.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yeekcay/Short-Video-CreatorX.git
   cd Short-Video-CreatorX
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **Critical for GPU Users**: The command above usually installs the CPU version of PyTorch by default. To enable GPU acceleration, you must uninstall `torch` and reinstall the CUDA-supported version.
   
   Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to generate the correct installation command for your system (e.g., Windows + CUDA 11.8).
   Example command:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

Run the script from the command line. The tool handles single files or entire folders.

### Basic Examples

**Caption a single video (Caption-Only Mode):**
```bash
python main.py --input video.mp4
```

**Caption with a Background Video (Overlay Mode):**
```bash
python main.py --input video.mp4 --background background.mp4
```

**Change Caption Position:**
```bash
python main.py --input video.mp4 --caption-position top
```
*Options: `top`, `center`, `bottom`*

**Batch Process a Folder:**
```bash
python main.py --input ./my_videos --output ./completed_videos
```

### CLI Arguments

| Argument | Description |
| :--- | :--- |
| `-i`, `--input` | Path to a video file or a folder of videos. |
| `-b`, `--background` | Path to a background video. If omitted, the script creates a caption-only video. |
| `-p`, `--caption-position` | Vertical position of the text (`top`, `center`, `bottom`). |
| `-o`, `--output` | Destination folder. Default is `OUTPUT_VIDEOS`. |

## Configuration

Settings are managed in `config.py`.

*   **Transcription:** Change `MODEL_NAME` (e.g., `base`, `small`, `large-v2`).
*   **Performance:** Toggle `USE_GPU_ENCODING` to `True` enables NVENC.
*   **Styling:** Adjust `FONT_SIZE`, `FONT_NAME`, and `SPEAKER_COLORS`.

### Enabling Speaker Diarization
To enable speaker detection (labelling who is speaking):

1.  Obtain a **Hugging Face Access Token**.
2.  Accept user agreements for `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1` on Hugging Face.
3.  Create a `.env` file in the project root:
    ```
    HUGGINGFACE_TOKEN=your_token_here
    ```
4.  Set `ENABLE_DIARIZATION = True` in `config.py`.

## troubleshooting

*   **CUDA/GPU Not Found:** Ensure your `torch` installation matches your CUDA version. Run `pip list` and check for `+cu` versions.
*   **OOM Errors:** If running out of VRAM, lower the `BATCH_SIZE` in `config.py` or switch to a smaller Whisper model.
*   **FFmpeg Error:** Ensure `ffmpeg` is accessible in your command line/terminal.

---
*Based on the original work by [sw-aka](https://github.com/sw-aka/Short-Video-Creator).*
