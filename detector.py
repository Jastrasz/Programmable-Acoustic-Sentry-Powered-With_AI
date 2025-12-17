import os
import sys
import time
import queue
import threading
import numpy as np
import cv2
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import config  # Import settings

# Try importing TensorFlow (Lite or Full)
try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except ImportError:
        print("Error: TensorFlow not found. Please install tflite_runtime or tensorflow.")
        sys.exit(1)

# Queue for inter-thread communication
file_queue = queue.Queue()

# Load AI Model
print(f"Loading AI Model: {config.MODEL_FILE}...")
try:
    interpreter = tflite.Interpreter(model_path=config.MODEL_FILE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("Make sure the .tflite file exists in the 'models' directory.")
    sys.exit(1)

def analyze_spectrogram(image_path):
    """Run inference on the generated spectrogram."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return 0.0
        
        # Preprocessing matching the training phase
        img = cv2.resize(img, config.IMAGE_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data[0][0] # Returns confidence score (0.0 - 1.0)
    except Exception as e:
        print(f"Inference Error: {e}")
        return 0.0

def create_spectrogram(audio_path, output_path, mic_index):
    try:
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
        D = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max, top_db=config.TOP_DB)

        fig, ax = plt.subplots()
        librosa.display.specshow(S_db, sr=sr, hop_length=config.HOP_LENGTH, 
                                 x_axis='time', y_axis='log', ax=ax, fmax=config.FMAX)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        
        base_name = os.path.basename(audio_path)
        img_filename = os.path.splitext(base_name)[0] + ".png"
        full_img_path = os.path.join(output_path, img_filename)

        plt.savefig(full_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # --- INFERENCE ---
        confidence = analyze_spectrogram(full_img_path)
        
        if confidence > config.DETECTION_THRESHOLD:
            print("\n" + "!!!" * 15)
            print(f" TARGET DETECTED! [Mic {mic_index}]")
            print(f" Confidence: {confidence*100:.1f}%")
            print("!!!" * 15 + "\n")
            # Here you can add code to trigger GPIO, send API request, etc.
        else:
            print(f"[Mic {mic_index}] Clear. (Conf: {confidence*100:.1f}%)")

        return full_img_path

    except Exception as e:
        print(f"Processing Error: {e}")
        return None

def worker_thread(output_path, mic_index):
    while True:
        try:
            audio_path = file_queue.get(timeout=1)
            if audio_path is None: break
            
            img_path = create_spectrogram(audio_path, output_path, mic_index)
            
            if config.CLEANUP_TEMP_FILES:
                try:
                    os.remove(audio_path)
                    if img_path: os.remove(img_path)
                except: pass
            
            file_queue.task_done()
        except queue.Empty:
            continue

def recorder_thread(mic_index, temp_path):
    print(f"Initializing Microphone: {mic_index}...")
    try:
        device_info = sd.query_devices(device=mic_index)
        channels = device_info['max_input_channels']
        if channels == 0: channels = config.CHANNELS_FALLBACK

        sd.check_input_settings(device=mic_index, channels=channels, samplerate=config.SAMPLE_RATE)
        sd.default.device = mic_index
        print(f"Mic {mic_index} Ready (Channels: {channels}).")
    except Exception as e:
        print(f"Mic Error: {e}")
        return

    while True:
        start_time = time.time()
        try:
            recording = sd.rec(int(config.CHUNK_DURATION * config.SAMPLE_RATE), 
                               samplerate=config.SAMPLE_RATE, channels=channels, dtype='int16')
            sd.wait()
            
            filename = f"rec_{int(start_time)}.wav"
            filepath = os.path.join(temp_path, filename)
            write(filepath, config.SAMPLE_RATE, recording)
            
            file_queue.put(filepath)
        except Exception as e:
            print(f"Recording Error: {e}")
        
        elapsed = time.time() - start_time
        time.sleep(max(0, config.CHUNK_DURATION - elapsed))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detector.py <MIC_INDEX>")
        sys.exit(1)
    
    try:
        MIC_INDEX = int(sys.argv[1])
    except ValueError:
        print("Error: Mic index must be an integer.")
        sys.exit(1)

    # Setup directories
    TEMP_DIR = f"./temp_audio/mic_{MIC_INDEX}/"
    SPEC_DIR = f"./temp_spec/mic_{MIC_INDEX}/"
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SPEC_DIR, exist_ok=True)

    print(f"--- ACOUSTIC SENTRY STARTED (Mic {MIC_INDEX}) ---")

    # Start Worker
    t_worker = threading.Thread(target=worker_thread, args=(SPEC_DIR, MIC_INDEX))
    t_worker.start()

    # Start Recorder (Main Thread)
    try:
        recorder_thread(MIC_INDEX, TEMP_DIR)
    except KeyboardInterrupt:
        print("Stopping...")
        file_queue.put(None)
        t_worker.join()