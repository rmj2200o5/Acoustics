# Acoustics

A Python GUI application for real-time audio analysis and synthesis.

## Features

### 🎤 Analyzer Tab
Records audio from the microphone, computes an FFT in real-time, and displays a **live frequency-spectrum plot**. The dominant (peak) frequency is highlighted below the chart.

### 🎹 Synthesizer Tab
Build a composite waveform by adding one or more **sine-wave components** (each with its own frequency, amplitude, and phase offset), preview the combined waveform, and play it through your speaker.

---

## How the App Works

### High-level architecture

```
┌──────────────────────────────────────────────────────────┐
│                     AcousticsApp (Tk)                    │
│  ┌───────────────────────┐  ┌──────────────────────────┐ │
│  │    AnalyzerFrame      │  │   SynthesizerFrame       │ │
│  │  (🎤 Analyzer tab)    │  │  (🎹 Synthesizer tab)    │ │
│  └───────────────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

The entry-point is `main()`, which creates one `AcousticsApp` window.  
`AcousticsApp` wraps a `ttk.Notebook` with two tabs — one for each major
feature — and applies a dark Catppuccin-style theme via `ttk.Style`.

---

### 🎤 Analyzer Tab (`AnalyzerFrame`)

**Goal:** capture live microphone audio, compute its frequency content, and
render an updating spectrum plot.

#### Data flow

```
Microphone
   │
   ▼  (sounddevice InputStream – background thread)
_audio_callback()
   │  rolls new samples into a ring buffer (_audio_buffer, 4096 samples)
   ▼
_update_plot()           ← called every 50 ms on the GUI thread via after()
   │  1. copies the buffer
   │  2. applies a Hann window  →  reduces spectral leakage
   │  3. rfft() + abs()         →  real-valued magnitude spectrum
   │  4. masks frequencies > 8 000 Hz
   │  5. normalizes to [0, 1]
   │  6. finds argmax (peak frequency, DC bin excluded)
   ▼
Matplotlib line plot (FigureCanvasTkAgg) + peak-frequency label
```

#### Key constants

| Constant | Value | Purpose |
|---|---|---|
| `SAMPLE_RATE` | 44 100 Hz | Standard CD-quality sample rate |
| `FFT_CHUNK` | 4 096 samples | FFT window size (≈93 ms at 44.1 kHz) |
| `MAX_FREQ_HZ` | 8 000 Hz | Highest frequency shown in the plot |
| `PLOT_INTERVAL_MS` | 50 ms | GUI refresh rate (~20 fps) |

#### Thread safety

`_audio_callback` runs in a `sounddevice` background thread.  
`_update_plot` runs on the Tkinter main thread.  
A `threading.Lock` (`_lock`) protects the shared `_audio_buffer`.

---

### 🎹 Synthesizer Tab (`SynthesizerFrame` + `SynthComponent`)

**Goal:** let the user compose a sound from one or more sine waves and play
it back.

#### Building a waveform

Each `SynthComponent` row exposes three parameters:

| Parameter | Range | Effect |
|---|---|---|
| Frequency (Hz) | 20 – 20 000 | Pitch of the sine wave |
| Amplitude | 0.00 – 1.00 | Loudness contribution of that partial |
| Phase (°) | 0 – 360 | Time offset of the sine wave |

`SynthesizerFrame._build_waveform(duration)` combines all active components:

```
signal(t) = Σ  amplitude_i · sin(2π · freq_i · t + phase_deg_i · π/180)
```

The result is then **peak-normalized** to `[-1, 1]` so the composite
waveform never clips regardless of how many components are added.

#### Waveform preview

Every time a component is added, removed, or its parameters change,
`_refresh_preview()` renders the first **50 ms** of the waveform in the
embedded Matplotlib plot so you can see the shape before playing.

#### Playback

Clicking **▶ Play** launches a daemon thread that calls `sd.play()` /
`sd.wait()` (from `sounddevice`), keeping the GUI responsive throughout.
When playback finishes (or the user clicks **⏹ Stop**), a callback posted
back to the GUI thread via `self.after(0, ...)` resets the button and
status label.

---

### Class summary

| Class | Role |
|---|---|
| `AcousticsApp` | Top-level `tk.Tk` window; owns the notebook and theme |
| `AnalyzerFrame` | Microphone capture, FFT, and live spectrum plot |
| `SynthComponent` | One sine-wave row (frequency + amplitude + phase controls) |
| `SynthesizerFrame` | Manages a list of `SynthComponent`s, builds and plays the composite waveform |

---

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
