# Acoustics

A Python GUI application for real-time audio analysis and synthesis.

## Features

### 🎤 Analyzer Tab
Records audio from the microphone, computes an FFT in real-time, and displays a **live frequency-spectrum plot**. The dominant (peak) frequency is highlighted below the chart. You can also capture the top frequency peaks and send them directly to the Synthesizer.

### 🎵 Tuner Tab
Detects the dominant pitch in real-time and displays how close it is to the nearest of the 12 western musical notes (A–G#). A visual **tuning meter** shows whether the pitch is sharp (red), flat (red), or in-tune (green). Perfect for tuning instruments or checking vocal pitch.

### 🎹 Synthesizer Tab
Build a composite waveform by adding one or more **sine-wave components** (each with its own frequency, amplitude, and phase offset), preview the combined waveform, play it through your speaker, and analyze it with the spectrogram tool.

Key features:
- **Quick-add preset notes** – Buttons for all 12 standard notes (C–B at A=440Hz) for quick frequency selection
- **Component harmonics** – Add harmonic series with realistic 1/n² amplitude scaling
- **Octave transposition** – Quickly raise or lower any component by one octave (▲ and ▼ buttons)
- **Sort components** – Organize by Frequency, Magnitude, or order added
- **Mouse wheel scrolling** – Scroll through components naturally with your mouse wheel
- **Spectrogram visualization** – Generate and view a spectrogram of the synthesized waveform

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

## How to Use

When the app launches you will see a window with three tabs: **Analyzer**, **Tuner**, and **Synthesizer**. The recording automatically stops when you switch between tabs.

### 🎤 Analyzer Tab

The Analyzer captures live audio from your microphone and shows a real-time frequency-spectrum plot.

1. **Start recording** – Click the **▶ Start Recording** button. The status label changes to *Recording…* and the spectrum plot begins updating (~20 times per second).
2. **Read the plot** – The horizontal axis shows frequency (0 – 4 000 Hz) and the vertical axis shows normalised magnitude (0 – 1). Taller peaks indicate stronger frequency components in the incoming audio.
3. **Peak frequency** – The dominant (loudest) frequency is displayed at the bottom of the tab in the **Peak frequency** field (e.g. `440.0 Hz`).
4. **Capture to Synthesizer** – Click **→ Synthesizer** to send the top N frequency peaks (configurable) to the Synthesizer with their magnitudes preserved.
5. **Stop recording** – Click **⏹ Stop Recording** to close the audio stream. The plot freezes and the peak-frequency display resets.

> **Tip:** Whistle, hum, or play a tone near the microphone and watch the corresponding peak appear on the spectrum.

### 🎵 Tuner Tab

The Tuner analyzes live audio and displays the dominant pitch relative to the nearest of the 12 western notes.

1. **Start recording** – Click **▶ Start Recording**. The tuner begins analyzing incoming audio.
2. **Read the meter** – A visual tuning meter shows:
   - **Green center line** – the reference pitch
   - **Red needle** – the detected pitch (moves left/right to show flat/sharp deviation)
   - **Cent deviation** – the exact number of cents (1 cent = 1/100 of a semitone) away from the nearest note
3. **Note display** – Shows which of the 12 notes is closest to the detected pitch (e.g., "A4 +5¢" = 5 cents sharp).
4. **Background noise filtering** – The tuner ignores very quiet background noise (–35 dB threshold), making it suitable for voice and guitar.
5. **Stop recording** – Click **⏹ Stop Recording** to stop analysis.

> **Tip:** Use this to tune a guitar string, check your vocal pitch, or calibrate other instruments. The tuner uses the A=440 Hz standard.

### 🎹 Synthesizer Tab

The Synthesizer lets you build a sound from one or more sine waves and play it through your speakers.

1. **Add a sine-wave component** – Click **➕ Add Sine Wave**. A new row appears with controls:
   - **Freq (Hz)** – the frequency of this sine wave (20 – 20 000 Hz, default 440 Hz / A4).
   - **Amplitude** – the relative loudness of this component (0.00 – 1.00, default 1.00).
   - **Phase (°)** – the phase offset in degrees (0 – 360, default 0).
   - **⋯ (More)** – menu button for additional options (currently: Add Harmonics).
   - **▲/▼ Octave buttons** – quickly transpose the frequency up or down by one octave.
   - **✕** – remove this component.

2. **Quick-add preset notes** – Use the buttons in the **Quick Add** frame (C through B) to instantly add any of the 12 standard western notes. All use A=440Hz as the reference.

3. **Add harmonics** – Click the **⋯** button on any component, then select "Add Harmonics". A dialog will prompt for the number of harmonics (1–10). Harmonics are generated with realistic 1/n² amplitude scaling (each harmonic decays in strength).

4. **Sort components** – Use the **Sort by** buttons to reorganize the component list:
   - **Freq** – sort by frequency (ascending)
   - **Mag** – sort by amplitude/magnitude (descending, loudest first)
   - **Order** – sort by the order components were added

5. **Scroll through components** – If you have many components, use your **mouse wheel** to scroll through the list while hovering over the components pane.

6. **Adjust values** – Use the spinner controls (click the ▲/▼ arrows or type a value directly) to set each parameter. The **Waveform Preview** plot at the bottom updates automatically to show the combined signal (first 50 ms).

7. **Set duration** – Use the **Duration (s)** spinner (top bar) to choose how many seconds of audio to generate (0.5 – 30 s, default 3 s).

8. **Generate spectrogram** – Click **📊 Spectrogram** to analyze the synthesized waveform and view its frequency content over time. A new window opens showing a spectrogram plot with frequency on the y-axis and time on the x-axis.

9. **Clear all components** – Click **🗑 Clear All** to remove all sine waves and start fresh.

10. **Play** – Click **▶ Play**. The waveform is generated and played through the default audio output. The button changes to **⏹ Stop**.

11. **Stop early** – Click **⏹ Stop** to interrupt playback before it finishes naturally.

> **Tip:** Add several components at different frequencies (e.g. 440 Hz, 880 Hz, 1 320 Hz) to hear a harmonic series, or combine closely-spaced frequencies to hear beating. Use the "Add Harmonics" feature to create rich, natural-sounding tones.
