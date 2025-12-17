# Configuration settings for Acoustic Sentry

# --- Audio Settings ---
SAMPLE_RATE = 44100
CHUNK_DURATION = 5  # Seconds
CHANNELS_FALLBACK = 2 # If auto-detection fails

# --- Spectrogram Settings ---
N_FFT = 4096
HOP_LENGTH = 256
TOP_DB = 60
FMAX = 20000
IMAGE_SIZE = (128, 128) # Input size for the AI model

# --- AI & Inference ---
MODEL_FILE = "models/model.tflite"
DETECTION_THRESHOLD = 0.8 # Confidence threshold (0.0 - 1.0)

# --- System ---
CLEANUP_TEMP_FILES = True
LOG_FILE = "detection_log.txt"