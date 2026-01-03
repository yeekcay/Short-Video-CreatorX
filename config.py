## Model settings
MODEL_NAME = 'whisper-small.en'
LANGUAGE = 'en'

## Processing settings
MAX_NUMBER_OF_PROCESSES = 1 # The maximum number of videos which can be processed simultaneously
NUM_THREADS = 8 # The number of threads used to save the editted video (reduce if using GPU encoding)

## GPU Encoding Settings
USE_GPU_ENCODING = True  # Set to True to use NVIDIA NVENC (requires compatible GPU and ffmpeg)
VIDEO_CODEC_NVENC = 'h264_nvenc'  # GPU hardware encoder
VIDEO_CODEC_CPU = 'libx264'  # CPU software encoder
ENCODING_PRESET = 'fast'  # Preset for encoding speed (p1-p7 for NVENC, ultrafast-veryslow for CPU)

## Font settings
FONT_NAME = 'Super Carnival.ttf' # The name of the font file used for captions
FONT_SIZE = 100
FONT_BORDER_WEIGHT = 10

## Video settings
USE_BACKGROUND_VIDEO = False # Set to False to caption videos without background footage
CAPTION_START_OFFSET = 0.05 # Delay in seconds (0.05 = 50ms) to prevent early captions due to breath/noise detection
FULL_RESOLUTION = (1080, 1920) # Resolution of the outputted video (width, height) in pixels
PERCENT_MAIN_CLIP = 40 # Percentage of output video height which is the main video (not the background video)
TEXT_POSITION_PERCENT = 30 # Position of caption text as a percentage of video height (from top of video)

## Source folders
INPUT_VIDEOS_DIR = 'INPUT_VIDEOS' # Directory of the input videos
OUTPUT_VIDEOS_DIR = 'OUTPUT_VIDEOS' # Directory the editted videos will be saved
BACKGROUND_VIDEOS_DIR = 'BACKGROUND_VIDEOS' # Directory of the background videos
FONTS_DIR = 'FONTS' # Directory the fonts are stored in

## WhisperX Settings
USE_WHISPERX = True  # Set to False to use legacy whisper-timestamped (fallback)
WHISPERX_MODEL = 'base'  # Options: 'tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'
COMPUTE_TYPE = 'float16'  # Options: 'float32' (CPU), 'float16' (GPU), 'int8' (GPU, lower quality but faster)
BATCH_SIZE = 16  # Reduce if running out of GPU memory (try 8, 4, or 2)

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

## Speaker Diarization Settings
ENABLE_DIARIZATION = True  # Enable speaker detection (requires HuggingFace token)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Get from .env file
SHOW_SPEAKER_LABELS = False  # Show "[Speaker 1]:" prefix in captions
MIN_SPEAKERS = 1  # Minimum number of speakers (set to 1 for single speaker, 2 for dialogue)
MAX_SPEAKERS = 2  # Maximum number of speakers (constrains the model to prevent over-segmentation)

## Speaker Color Mapping (used when ENABLE_DIARIZATION = True)
SPEAKER_COLORS = {
    'SPEAKER_00': '#FFFFFF',  # White
    'SPEAKER_01': '#FFD700',  # Gold
    'SPEAKER_02': '#00FFFF',  # Cyan
    'SPEAKER_03': '#FF69B4',  # Hot Pink
    'SPEAKER_04': '#7FFF00',  # Chartreuse
    'SPEAKER_05': '#FF6347',  # Tomato
    'SPEAKER_06': '#9370DB',  # Medium Purple
    'SPEAKER_07': '#00FA9A',  # Medium Spring Green
}
DEFAULT_SPEAKER_COLOR = '#FFFFFF'  # Fallback color for unknown speakers
