# Acoustic Sentry: Edge AI Audio Detection

**A universal, privacy-focused acoustic monitoring system for Raspberry Pi.**

This project allows you to turn a Raspberry Pi and a set of USB microphones into an intelligent acoustic sensor network. It uses Deep Learning (CNNs) to classify audio events in real-time.

While originally designed for **Drone Detection**, the architecture is completely agnostic and can be trained to detect:
* 👶 Baby crying
* 🪟 Breaking glass
* 🐶 Dog barking
* 🚨 Sirens or Alarms
* 🏭 Machinery failure patterns

## Architecture

1.  **Capture**: Records continuous audio chunks (default: 5s) via ALSA/SoundDevice.
2.  **Transform**: Converts raw audio into Mel-Spectrograms (images) using Librosa.
3.  **Inference**: Analyzes the spectrogram using a custom-trained TensorFlow Lite model.
4.  **Alert**: Triggers an event if the confidence score exceeds the threshold.

## Hardware Requirements

* **Raspberry Pi** (3B, 4, or 5 recommended)
* **USB Microphones** (Supports multi-microphone arrays via USB Hub)
* Python 3.7+

## Installation

1.  Clone the repository:
    ```bash
    git clone (https://github.com/Jastrasz/Acoustic-Sentry.git)
    cd Acoustic-Sentry
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: On Raspberry Pi, you might need to install `libatlas-base-dev` and `portaudio19-dev` via apt).*

## Usage

1.  Place your trained `.tflite` model in the `models/` directory and update `config.py`.
2.  Run the detector specifying the microphone index (find indices via `arecord -l`):
3. For example:
    ```bash
    python detector.py 2
    ```

## Training Your Own Model

1.  Collect audio samples for your **Target** class (e.g., drones) and **Background** class (noise).
2.  Place them in `dataset/target/` and `dataset/background/`.
3.  Run the training script (coming soon / see `train.py`).
4.  Convert the model to TFLite and transfer to the Pi.

## License

This project is licensed under the MIT License - see the LICENSE file for details.