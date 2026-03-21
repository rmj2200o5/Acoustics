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
