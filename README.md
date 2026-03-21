# Acoustics

A Python GUI application for real-time audio analysis and synthesis.

## Features

### 🎤 Analyzer Tab
Records audio from the microphone, computes an FFT in real-time, and displays a **live frequency-spectrum plot**. The dominant (peak) frequency is highlighted below the chart.

### 🎹 Synthesizer Tab
Build a composite waveform by adding one or more **sine-wave components** (each with its own frequency, amplitude, and phase offset), preview the combined waveform, and play it through your speaker.

## Requirements

- Python 3.10+
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- PortAudio (system library required by sounddevice)

### Installing system dependencies

```bash
# Ubuntu / Debian
sudo apt-get install python3-tk libportaudio2
```

### Installing Python dependencies

```bash
pip install -r requirements.txt
```

## Running the application

```bash
python acoustics_app.py
```

## How to Use

When the app launches you will see a window with two tabs: **Analyzer** and **Synthesizer**.

### 🎤 Analyzer Tab

The Analyzer captures live audio from your microphone and shows a real-time frequency-spectrum plot.

1. **Start recording** – Click the **▶ Start Recording** button. The status label changes to *Recording…* and the spectrum plot begins updating (~20 times per second).
2. **Read the plot** – The horizontal axis shows frequency (0 – 8 000 Hz) and the vertical axis shows normalised magnitude (0 – 1). Taller peaks indicate stronger frequency components in the incoming audio.
3. **Peak frequency** – The dominant (loudest) frequency is displayed at the bottom of the tab in the **Peak frequency** field (e.g. `440.0 Hz`).
4. **Stop recording** – Click **⏹ Stop Recording** to close the audio stream. The plot freezes and the peak-frequency display resets.

> **Tip:** Whistle, hum, or play a tone near the microphone and watch the corresponding peak appear on the spectrum.

### 🎹 Synthesizer Tab

The Synthesizer lets you build a sound from one or more sine waves and play it through your speakers.

1. **Add a sine-wave component** – Click **➕ Add Sine Wave**. A new row appears with three controls:
   - **Freq (Hz)** – the frequency of this sine wave (20 – 20 000 Hz, default 440 Hz / A4).
   - **Amplitude** – the relative loudness of this component (0.00 – 1.00, default 1.00).
   - **Phase (°)** – the phase offset in degrees (0 – 360, default 0).
2. **Adjust values** – Use the spinner controls (click the ▲/▼ arrows or type a value directly) to set each parameter. The **Waveform Preview** plot at the bottom updates automatically to show the combined signal (first 50 ms).
3. **Remove a component** – Click the **✕** button on any row to remove that sine wave.
4. **Set duration** – Use the **Duration (s)** spinner (top bar) to choose how many seconds of audio to generate (0.5 – 30 s, default 3 s).
5. **Play** – Click **▶ Play**. The waveform is generated and played through the default audio output. The button changes to **⏹ Stop**.
6. **Stop early** – Click **⏹ Stop** to interrupt playback before it finishes naturally.

> **Tip:** Add several components at different frequencies (e.g. 440 Hz, 880 Hz, 1 320 Hz) to hear a harmonic series, or combine closely-spaced frequencies to hear beating.
