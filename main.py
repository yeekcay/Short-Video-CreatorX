import math
import multiprocessing
import os
import random
import shutil
import time
import logging
import subprocess
import argparse


import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (VideoFileClip, clips_array, concatenate_videoclips,
                             ImageClip, CompositeVideoClip, VideoClip)
from moviepy.video.fx.all import crop as moviepy_crop
import whisperx
from whisperx.diarize import DiarizationPipeline

from config import (
    BACKGROUND_VIDEOS_DIR,
    FONT_BORDER_WEIGHT,
    FONTS_DIR,
    FONT_NAME,
    FONT_SIZE,
    FULL_RESOLUTION,
    INPUT_VIDEOS_DIR,
    MAX_NUMBER_OF_PROCESSES,
    OUTPUT_VIDEOS_DIR,
    PERCENT_MAIN_CLIP,
    TEXT_POSITION_PERCENT,
    LANGUAGE,
    NUM_THREADS,
    USE_BACKGROUND_VIDEO,
    CAPTION_START_OFFSET,
    # WhisperX settings
    WHISPERX_MODEL,
    COMPUTE_TYPE,
    BATCH_SIZE,
    ENABLE_DIARIZATION,
    HUGGINGFACE_TOKEN,
    SHOW_SPEAKER_LABELS,
    MIN_SPEAKERS,
    MAX_SPEAKERS,
    SPEAKER_COLORS,
    DEFAULT_SPEAKER_COLOR,
    USE_GPU_ENCODING,
    VIDEO_CODEC_NVENC,
    VIDEO_CODEC_CPU,
    ENCODING_PRESET
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper Functions
def detect_device():
    """Detect if CUDA GPU is available, otherwise use CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            logging.info("CUDA GPU detected - using GPU acceleration")
            return "cuda"
        else:
            logging.info("No CUDA GPU detected - using CPU")
            return "cpu"
    except ImportError:
        logging.warning("PyTorch not found - defaulting to CPU")
        return "cpu"

def get_speaker_color(speaker_id):
    """Get color for a speaker ID, with fallback to default color."""
    if speaker_id is None:
        return DEFAULT_SPEAKER_COLOR
    return SPEAKER_COLORS.get(speaker_id, DEFAULT_SPEAKER_COLOR)

class VideoTools:
    # Initialize the VideoTools class with a VideoFileClip
    clip: VideoFileClip = None

    def __init__(self, clip: VideoFileClip) -> None:
        """Constructor to initialize the VideoFileClip."""
        self.clip = clip

    def __deinit__(self) -> None:
        """Destructor to clean up resources."""
        if self.clip:
            self.clip.close()  # Close the clip to free resources
            self.clip = None  # Set clip to None to avoid dangling reference

    def crop(self, width: int, height: int) -> VideoFileClip:
        """Crop the video clip to the specified width and height.

        Args:
            width (int): The desired width of the cropped video.
            height (int): The desired height of the cropped video.

        Returns:
            VideoFileClip: The cropped video clip.
        """
        # Get the original dimensions of the video clip
        original_width, original_height = self.clip.size

        # Calculate the change ratios for width and height
        width_change_ratio = width / original_width
        height_change_ratio = height / original_height

        # Determine the maximum ratio to maintain aspect ratio
        max_ratio = max(width_change_ratio, height_change_ratio)

        # Resize the clip based on the maximum ratio
        self.clip = self.clip.resize((
            original_width * max_ratio,
            original_height * max_ratio,
        ))

        # Get the new dimensions after resizing
        new_width, new_height = self.clip.size

        # Crop the video based on the aspect ratio
        if width_change_ratio > height_change_ratio:
            # Calculate the vertical crop
            height_change = new_height - height
            new_y1 = round(height_change / 2)  # Calculate the starting y-coordinate
            new_y2 = min(new_y1 + height, new_height)  # Calculate the ending y-coordinate
            self.clip = moviepy_crop(self.clip, y1=new_y1, y2=new_y2)  # Crop the video
        elif height_change_ratio > width_change_ratio:
            # Calculate the horizontal crop
            width_change = new_width - width
            new_x1 = round(width_change / 2)  # Calculate the starting x-coordinate
            new_x2 = min(new_x1 + width, new_width)  # Calculate the ending x-coordinate
            self.clip = moviepy_crop(self.clip, x1=new_x1, x2=new_x2)  # Crop the video
            self.clip = self.clip.resize((width, height))  # Resize to the final dimensions

        return self.clip  # Return the cropped video clip


class Tools:
    @staticmethod
    def round_down(num: float, decimals: int = 0) -> float:
        """
        Rounds down a number to a specified number of decimal places.

        :param num: The number to round down.
        :param decimals: The number of decimal places to round to (default is 0).
        :return: The rounded down number.
        """
        return math.floor(num * 10 ** decimals) / 10 ** decimals

class BackgroudVideo:
    @staticmethod
    def get_clip(duration: float, background_path: str = None) -> VideoFileClip:
        """
        Retrieves a background video clip, trims it to the specified duration,
        and crops it to the target resolution.

        :param duration: The desired duration of the video clip.
        :param background_path: Optional path to a specific background video. If None, selects randomly.
        :return: A cropped and trimmed VideoFileClip object.
        """
        # Select a clip - either specified or random
        if background_path:
            full_clip = VideoFileClip(background_path)
        else:
            full_clip = VideoFileClip(BackgroudVideo.select_clip())
        
        # Trim the selected clip to the specified duration
        trimmed_clip = BackgroudVideo.trim_clip(full_clip, duration)

        # Crop the trimmed clip to 90% of its width
        width, height = trimmed_clip.size
        trimmed_clip = VideoTools(trimmed_clip).crop(round(width * 0.9), height)

        # Get the target resolution for the final clip
        target_resolution = BackgroudVideo.get_target_resolution()
        
        # Crop the trimmed clip to the target resolution
        cropped_clip = VideoTools(trimmed_clip).crop(target_resolution[0], target_resolution[1])

        # Return the cropped clip without audio
        return cropped_clip.set_audio(None)
    
    @staticmethod
    def select_clip() -> str:
        """
        Selects a random video clip from the background videos directory.

        :return: The file path of the selected video clip.
        """
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
        clip = random.choice(clips)
        return os.path.join(BACKGROUND_VIDEOS_DIR, clip)
    
    @staticmethod
    def trim_clip(clip: VideoFileClip, duration: float) -> VideoFileClip:
        """
        Trims a video clip to a specified duration.

        :param clip: The VideoFileClip to trim.
        :param duration: The desired duration of the trimmed clip.
        :return: A trimmed VideoFileClip object.
        :raises ValueError: If the clip's duration is less than the specified duration.
        """
        if clip.duration < duration:
            raise ValueError(f"Clip duration {clip.duration} is less than duration {duration}")
        
        # Randomly select a start time for the subclip
        clip_start_time = Tools.round_down(random.uniform(0, clip.duration - duration))
        return clip.subclip(clip_start_time, clip_start_time + duration)

    @staticmethod
    def get_target_resolution():
        """
        Calculates the target resolution for the video clip based on the full resolution
        and the percentage reduction for the main clip.

        :return: A tuple containing the target width and height.
        """
        return (
            FULL_RESOLUTION[0], 
            round(FULL_RESOLUTION[1] * (1 - (PERCENT_MAIN_CLIP / 100)))
        )
    
    @staticmethod
    def format_all_background_clips():
        """
        Formats all background video clips in the specified directory by cropping them
        to the full resolution and saving them back to the directory.

        :return: None
        """
        clips = os.listdir(BACKGROUND_VIDEOS_DIR)
        for clip_name in clips:
            # Load each clip and crop it to the full resolution
            clip = VideoFileClip(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name))
            clip = VideoTools(clip).crop(FULL_RESOLUTION[0], FULL_RESOLUTION[1])

            # Save the formatted clip back to the directory
            clip.write_videofile(os.path.join(BACKGROUND_VIDEOS_DIR, clip_name), codec="libx264", audio_codec="aac")
            
class VideoCreation:
    # Class attributes for video and audio clips
    clip = None
    audio = None
    background_clip = None
    background_path = None
    caption_position = None

    def __init__(self, clip: VideoFileClip, background_path: str = None, caption_position: str = None) -> None:
        # Initialize the VideoCreation object with a video clip
        self.clip = clip
        self.audio = clip.audio  # Extract audio from the video clip
        self.background_path = background_path  # Store background path for later use
        self.caption_position = caption_position  # Store caption position for later use

    def __deinit__(self) -> None:
        # Clean up resources by closing video and background clips
        if self.clip:
            self.clip.close()
            self.clip = None
        if self.background_clip:
            self.background_clip.close()
            self.background_clip = None

    def process(self) -> VideoClip:
        # Main processing function to create the final video
        self.clip = self.create_final_clip()  # Create the final video clip
        transcription = self.create_transcription(self.audio)  # Generate transcription from audio
        self.clip = self.add_captions_to_video(self.clip, transcription)  # Add captions to the video

        return self.clip  # Return the processed video clip

    def create_final_clip(self):
        # Create the final video clip with or without a background
        if self.background_path or USE_BACKGROUND_VIDEO:
            # Background mode: combine with background video
            self.background_clip = BackgroudVideo.get_clip(self.clip.duration, self.background_path)  # Get background video clip

            _, background_height = self.background_clip.size  # Get the height of the background clip
            target_dimensions = (FULL_RESOLUTION[0], FULL_RESOLUTION[1] - background_height)  # Calculate target dimensions
            self.clip = VideoTools(self.clip).crop(target_dimensions[0], target_dimensions[1])  # Crop the main clip

            # Combine the main clip and background clip
            self.clip = clips_array([[self.clip], [self.background_clip]])
        else:
            # Caption-only mode: just crop the main video to full resolution
            self.clip = VideoTools(self.clip).crop(FULL_RESOLUTION[0], FULL_RESOLUTION[1])
        
        return self.clip  # Return the final clip

    def create_transcription(self, audio):
        """Generate transcription using WhisperX 3-stage pipeline."""
        os.makedirs("temp", exist_ok=True)
        
        # Create a unique file name for the audio file
        file_dir = f"temp/{time.time() * 10**20:.0f}.mp3"
        audio.write_audiofile(file_dir, codec="mp3", verbose=False, logger=None)
        
        # Wait until the audio file is created
        while not os.path.exists(file_dir):
            time.sleep(0.01)
        
        try:
            # Detect device (GPU or CPU)
            device = detect_device()
            
            # Adjust compute type based on device
            compute_type = COMPUTE_TYPE
            if device == "cpu" and compute_type in ["float16", "int8"]:
                logging.warning(f"Compute type '{compute_type}' not supported on CPU, using 'float32'")
                compute_type = "float32"
            
            logging.info(f"Loading WhisperX model: {WHISPERX_MODEL}")
            
            # STAGE 1: Transcription with faster-whisper
            model = whisperx.load_model(
                WHISPERX_MODEL, 
                device=device, 
                compute_type=compute_type,
                language=LANGUAGE
            )
            
            # Load audio
            audio_data = whisperx.load_audio(file_dir)
            
            logging.info("Stage 1: Transcribing audio...")
            result = model.transcribe(audio_data, batch_size=BATCH_SIZE)
            
            # Clean up model from memory if needed
            del model
            import gc
            gc.collect()
            
            # STAGE 2: Alignment for precise word-level timestamps
            logging.info("Stage 2: Aligning timestamps...")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], 
                device=device
            )
            
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio_data, 
                device,
                return_char_alignments=False
            )
            
            # Clean up alignment model
            del model_a
            gc.collect()
            
            # STAGE 3: Speaker Diarization (optional)
            speaker_segments = None
            diarization_success = False  # Track if diarization completed successfully
            
            if ENABLE_DIARIZATION and HUGGINGFACE_TOKEN:
                try:
                    logging.info("Stage 3: Identifying speakers...")
                    diarize_model = DiarizationPipeline(
                        use_auth_token=HUGGINGFACE_TOKEN, 
                        device=device
                    )
                    
                    # Run diarization
                    diarize_kwargs = {}
                    if MIN_SPEAKERS is not None:
                        diarize_kwargs['min_speakers'] = MIN_SPEAKERS
                    if MAX_SPEAKERS is not None:
                        diarize_kwargs['max_speakers'] = MAX_SPEAKERS
                    
                    diarize_segments = diarize_model(audio_data, **diarize_kwargs)
                    
                    # Assign speakers to words
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    speaker_segments = diarize_segments
                    
                    # Count unique speakers from the result (after assignment)
                    unique_speakers = set()
                    for segment in result.get("segments", []):
                        for word in segment.get("words", []):
                            if "speaker" in word:
                                unique_speakers.add(word["speaker"])
                    
                    if unique_speakers:
                        logging.info(f"Detected {len(unique_speakers)} speaker(s): {', '.join(sorted(unique_speakers))}")
                        diarization_success = True  # Mark as successful
                    else:
                        logging.warning("Speaker diarization completed but no speakers were assigned to words")
                    
                    del diarize_model
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Speaker diarization failed: {type(e).__name__}: {e}")
                    logging.info("Continuing without speaker detection...")
                    diarization_success = False
            
            # Extract timestamps and words from the result
            timestamps = []
            
            for segment in result["segments"]:
                if "words" not in segment:
                    continue
                    
                for word_info in segment["words"]:
                    # WhisperX alignment is more accurate, so we don't need CAPTION_START_OFFSET
                    # But we can still apply it if configured
                    start_time = word_info.get('start', 0)
                    end_time = word_info.get('end', start_time + 0.5)
                    text = word_info.get('word', '').strip()
                    # Only use speaker data if diarization completed successfully
                    speaker = word_info.get('speaker', None) if diarization_success else None
                    
                    if not text:
                        continue
                    
                    # Optional: Apply caption start offset if still needed
                    if CAPTION_START_OFFSET > 0:
                        new_start = start_time + CAPTION_START_OFFSET
                        if new_start < end_time:
                            start_time = new_start
                    
                    timestamps.append({
                        'timestamp': (start_time, end_time),
                        'text': text,
                        'speaker': speaker
                    })
            
            logging.info(f"Transcription complete: {len(timestamps)} words")
            
        finally:
            # Clean up the temporary audio file
            try:
                os.remove(file_dir)
            except FileNotFoundError:
                pass
        
        return timestamps

    def add_captions_to_video(self, clip, timestamps):
        # Add captions to the video based on the provided timestamps
        if len(timestamps) == 0:
            return clip  # Return the original clip if no timestamps

        clips = []  # List to hold video clips with captions
        previous_time = 0  # Track the end time of the previous caption

        queued_texts = []  # List to hold texts for the current caption
        full_start = None  # Start time for the current caption

        end = 0  # End time for the current caption

        # Iterate through the timestamps to create captions
        for pos, timestamp in enumerate(timestamps):
            start, end = timestamp["timestamp"]
            text = timestamp["text"]
            speaker = timestamp.get("speaker", None)  # Get speaker if available

            # If there is a gap before the current caption, add the previous clip
            if start > previous_time and len(queued_texts) == 0:
                clips.append(clip.subclip(previous_time, start))

            # Adjust the end time if there is a next timestamp
            if pos + 1 < len(timestamps):
                next_timestamp_start = timestamps[pos + 1]['timestamp'][0]
                if next_timestamp_start > end:
                    if next_timestamp_start - end > 0.5:
                        end += 0.5
                    else:
                        end = next_timestamp_start

            # If the gap between captions is small, queue the text
            if end - previous_time < 0.3 and pos + 1 < len(timestamps):
                if full_start is None:
                    full_start = start
                queued_texts.append(text)
                continue

            queued_texts.append(text)  # Add the current text to the queue

            # Combine queued texts into a single caption
            if len(queued_texts) > 0:
                text = " ".join(queued_texts)
                queued_texts = []

            if full_start is None:
                full_start = start

            # Skip if the caption exceeds the clip duration
            if full_start > clip.duration or end > clip.duration:
                continue

            # Add the captioned clip to the list
            clips.append(
                self.add_text_to_video(
                    clip.subclip(full_start, end),
                    text,
                    speaker  # Pass speaker info
                )
            )

            previous_time = end  # Update the previous time
            full_start = None  # Reset full start for the next caption

        # Add any remaining clip after the last caption
        if clip.duration - end > 0.01:
            clips.append(
                clip.subclip(end, clip.duration)
            )

        clip = concatenate_videoclips(clips)  # Concatenate all clips with captions

        return clip  # Return the final clip with captions

    def add_text_to_video(self, clip, text, speaker=None):
        """Add text overlay to the video clip with optional speaker color."""
        text_image = self.create_text_image(
            text,
            os.path.join(FONTS_DIR, FONT_NAME),
            FONT_SIZE,
            clip.size[0],
            speaker  # Pass speaker for color selection
        )

        image_clip = ImageClip(np.array(text_image), duration=clip.duration)  # Create an image clip for the text

        # Calculate vertical position for text based on caption_position
        if self.caption_position == 'top':
            y_offset = round(FULL_RESOLUTION[1] * 0.30)
        elif self.caption_position == 'center':
            y_offset = round(FULL_RESOLUTION[1] * 0.50)
        elif self.caption_position == 'bottom':
            y_offset = round(FULL_RESOLUTION[1] * 0.70)
        else:
            # Smart default: bottom for caption-only, top for combined mode
            if self.background_path or USE_BACKGROUND_VIDEO:
                y_offset = round(FULL_RESOLUTION[1] * 0.30)  # Top for combined mode
            else:
                y_offset = round(FULL_RESOLUTION[1] * 0.70)  # Bottom for caption-only
        
        clip = CompositeVideoClip([clip, image_clip.set_position((0, y_offset,))])  # Overlay text on the video

        return clip  # Return the video clip with text

    def create_text_image(self, text, font_path, font_size, max_width, speaker=None):
        """Create an image with the specified text and speaker-based color."""
        # Add speaker label if enabled
        if speaker and SHOW_SPEAKER_LABELS:
            # Format speaker ID nicely (e.g., "SPEAKER_00" -> "Speaker 1")
            speaker_num = int(speaker.split('_')[-1]) + 1 if '_' in speaker else 1
            text = f"[Speaker {speaker_num}]: {text}"
        
        # Get color for this speaker
        text_color = get_speaker_color(speaker)
        
        # Create an image with the specified text
        image = Image.new("RGBA", (max_width, font_size * 10), (0, 0, 0, 0))

        font = ImageFont.truetype(font_path, font_size)

        draw = ImageDraw.Draw(image)

        # Get the bounding box for the text
        _, _, w, h = draw.textbbox((0, 0), text, font=font)

        # Draw the text on the image with stroke for better visibility
        draw.text(
            ((max_width - w) / 2, round(h * 0.2)), 
            text, 
            font=font, 
            fill=text_color,  # Use speaker-based color
            stroke_width=FONT_BORDER_WEIGHT, 
            stroke_fill='black'
        )

        image = image.crop((0, 0, max_width, round(h * 1.6),))  # Crop the image to the desired size

        return image  # Return the created text image




def start_process(file_name, processes_status_dict, video_queue: multiprocessing.Queue, background_path: str = None, caption_position: str = None, output_folder: str = None):
    """
    Process a video file by applying transformations and saving the output.

    Args:
        file_name (str): The name of the video file to process.
        processes_status_dict (dict): A dictionary to track the status of processes.
        video_queue (multiprocessing.Queue): A queue to manage video processing tasks.
        background_path (str): Optional path to background video file.
        caption_position (str): Optional caption position (top/center/bottom).
        output_folder (str): Optional output folder path.
    """
    
    logging.info(f"Processing: {file_name}")  # Log the start of processing
    start_time = time.time()  # Record the start time

    # Get the current process identifier
    process_identifier = multiprocessing.current_process().pid

    # Mark the process as not finished in the status dictionary
    processes_status_dict[process_identifier] = False

    # Load the input video file
    if os.path.exists(file_name):
        input_path = file_name
    else:
        input_path = os.path.join(INPUT_VIDEOS_DIR, file_name)

    input_video = VideoFileClip(input_path)
    
    # Process the video using a custom VideoCreation class
    output_video = VideoCreation(input_video, background_path, caption_position).process()
    
    
    logging.info(f"Saving: {file_name}")  # Log the saving process

    # Determine output folder (use custom or default)
    final_output_folder = output_folder if output_folder else OUTPUT_VIDEOS_DIR
    os.makedirs(final_output_folder, exist_ok=True)
    
    # Generate unique filename with numbered suffix if file exists
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    extension = os.path.splitext(file_name)[1]
    
    # Add _output suffix
    base_name = f"{base_name}_output"
    
    output_dir = os.path.join(final_output_folder, f"{base_name}{extension}")
    
    # If file exists, add numbered suffix (_1, _2, _3, etc.)
    counter = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(final_output_folder, f"{base_name}_{counter}{extension}")
        counter += 1
    
    end_time = round(((output_video.duration * 100 // output_video.fps) * output_video.fps / 100), 2)
    
    # Create a subclip of the output video
    output_video = output_video.subclip(t_end=end_time)

    # Determine codec and preset
    video_codec = VIDEO_CODEC_CPU
    encoding_preset = ENCODING_PRESET
    output_threads = NUM_THREADS
    
    if USE_GPU_ENCODING:
        try:
            import torch
            if torch.cuda.is_available():
                video_codec = VIDEO_CODEC_NVENC
                output_threads = None
                logging.info("Using GPU encoding (NVENC)")
        except:
            pass

    # Attempt to save the output video, retrying up to 5 times on failure
    for pos in range(5):
        try:
            output_video.write_videofile(
                output_dir,
                codec=video_codec,
                preset=encoding_preset,
                audio_codec="aac",
                fps=output_video.fps,
                threads=output_threads,
                verbose=False,
                logger=None
            )
            break  # Exit the loop if saving is successful
        except IOError:
            logging.warning(f"ERROR Saving: {file_name}. Trying again {pos + 1}/5")  # Log the error and retry
            time.sleep(1)  # Wait before retrying
    else:
        logging.error(f"ERROR Saving: {file_name}")  # Log if all attempts failed
    
    # Close the input and output video files to free resources
    input_video.close()
    output_video.close()

    # Log the runtime of the processing
    logging.info(f"Runtime: {round(time.time() - start_time, 2)} - {file_name}")
    
    # Mark the process as finished in the status dictionary
    processes_status_dict[process_identifier] = True


def delete_temp_folder():
    """
    Delete the temporary folder used for processing videos.
    """
    try:
        shutil.rmtree('temp')  # Remove the 'temp' directory and all its contents
    except (PermissionError, FileNotFoundError):
        pass  # Ignore permission errors if the folder cannot be deleted or is not found


import subprocess


    


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='AI Short Video Creator - Add captions to videos')
    parser.add_argument('--input', '-i', type=str, 
                        help='Path to input video file. If not specified, processes all videos in INPUT_VIDEOS folder.')
    parser.add_argument('--background', '-b', type=str, 
                        help='Path to background video file. If not specified, caption-only mode.')
    parser.add_argument('--caption-position', '-p', type=str, 
                        choices=['top', 'center', 'bottom'],
                        help='Caption position. Default: bottom for caption-only, top for combined mode.')
    parser.add_argument('--output', '-o', type=str,
                        help='Output folder path. Default: OUTPUT_VIDEOS folder.')
    args = parser.parse_args()
    
    # Clean up any temporary folders before starting
    delete_temp_folder()
    
    # Create a manager for shared data between processes
    manager = multiprocessing.Manager()
    processes_status_dict = manager.dict()  # Dictionary to track process statuses
    video_queue = multiprocessing.Queue()    # Queue to hold video file names

    # Create input and output directories if they don't exist
    os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

    # Determine which videos to process based on --input argument
    # Determine which videos to process based on --input argument
    if args.input:
        # Check if input is a directory or file
        if os.path.isdir(args.input):
            # Process all valid video files in the directory
            files = os.listdir(args.input)
            valid_extensions = ('.mp4', '.mov', '.avi', '.mkv')
            input_video_names = [os.path.join(args.input, f) for f in files if f.lower().endswith(valid_extensions)]
            
            if not input_video_names:
                logging.error(f"No video files found in directory: {args.input}")
                exit(1)
                
        elif os.path.exists(args.input):
            # Process specific video file using provided path
            input_video_names = [args.input]
            
        elif os.path.exists(os.path.join(INPUT_VIDEOS_DIR, args.input)):
            # Check if it's in INPUT_VIDEOS folder
            input_video_names = [args.input]
        else:
            logging.error(f"Input video or directory not found: {args.input}")
            exit(1)
    else:
        # Process all videos in INPUT_VIDEOS folder (default)
        input_video_names = os.listdir(INPUT_VIDEOS_DIR)

    # Add video file names to the queue
    for name in input_video_names:
        video_queue.put(name)

    processes = {} # Dictionary to store processes
    num_active_processes = 0  # Counter for active processes
    logging.info('STARTED')

    # Main loop to manage video processing
    while (video_queue.qsize() != 0) or (len(processes) != 0):
        # Check if we can start a new process
        if (num_active_processes < MAX_NUMBER_OF_PROCESSES) and (video_queue.qsize() > 0):
            file_name = video_queue.get()  # Get the next video file name from the queue

            # Create a new process for video processing
            p = multiprocessing.Process(target=start_process, args=(file_name, processes_status_dict, video_queue, args.background, args.caption_position, args.output))
            p.start()  # Start the process
            processes[p.pid] = p  # Store the process in the dictionary
            num_active_processes += 1  # Increment the active process counter

        # Check for completed processes
        for pid, complete in processes_status_dict.items():
            if complete:  # If the process is complete
                processes[pid].join()  # Wait for the process to finish
                del processes[pid]  # Remove the process from the dictionary
                del processes_status_dict[pid]  # Remove the status from the dictionary
                num_active_processes -= 1  # Decrement the active process counter

    # Clean up temporary folders after processing is complete
    delete_temp_folder()
    logging.info('MAIN PROCESS COMPLETE')






