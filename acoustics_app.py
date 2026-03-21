"""
Acoustics Application
=====================
A GUI application with three main features:
  1. Analyzer – records audio from the microphone, performs a real-time FFT
     and displays a live frequency-spectrum plot.
  2. Tuner – detects the dominant pitch and shows how close it is to the nearest
     musical note (using A=440Hz standard) with visual green/red feedback.
  3. Synthesizer – lets the user build a waveform from one or more sine-wave
     components and play the result through the default speaker.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100          # Hz
FFT_CHUNK   = 4096           # samples per FFT window
MAX_FREQ_HZ = 4000           # highest frequency shown on the spectrum plot
PLOT_INTERVAL_MS = 50        # GUI refresh rate (ms)  → ~20 fps


class AnalyzerFrame(ttk.Frame):
    """
    Tab 1 – Microphone → FFT → Live frequency-spectrum plot.
    """

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._audio_buffer = np.zeros(FFT_CHUNK, dtype=np.float32)
        self._last_freqs = np.array([])
        self._last_spectrum = np.array([])
        self._synthesizer_frame = None  # Will be set by parent app
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        control_bar = ttk.Frame(self)
        control_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self._btn_toggle = ttk.Button(
            control_bar, text="▶  Start Recording", command=self._toggle_recording
        )
        self._btn_toggle.pack(side=tk.LEFT, padx=4)

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(control_bar, textvariable=self._status_var).pack(side=tk.LEFT, padx=12)

        # Frequency capture controls
        ttk.Separator(control_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(control_bar, text="Capture top").pack(side=tk.LEFT, padx=(4, 2))
        self._capture_count_var = tk.IntVar(value=5)
        ttk.Spinbox(
            control_bar, from_=1, to=20, textvariable=self._capture_count_var, width=4
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(control_bar, text="frequencies").pack(side=tk.LEFT, padx=2)

        self._btn_capture = ttk.Button(
            control_bar, text="→ Synthesizer", command=self._capture_to_synthesizer, state=tk.DISABLED
        )
        self._btn_capture.pack(side=tk.LEFT, padx=4)

        # ── Matplotlib figure ──────────────────────────────────────────────────
        fig = Figure(figsize=(9, 4), dpi=96, facecolor="#1e1e2e")
        self._ax = fig.add_subplot(111)
        self._setup_axes()

        freqs = np.linspace(0, MAX_FREQ_HZ, FFT_CHUNK // 2)
        (self._line,) = self._ax.plot(freqs, np.zeros(FFT_CHUNK // 2), color="#89b4fa", lw=1.2)

        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # ── Fundamental frequency display ──────────────────────────────────────
        info_bar = ttk.Frame(self)
        info_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        ttk.Label(info_bar, text="Peak frequency:").pack(side=tk.LEFT)
        self._peak_var = tk.StringVar(value="–")
        ttk.Label(info_bar, textvariable=self._peak_var, font=("Helvetica", 12, "bold")).pack(
            side=tk.LEFT, padx=6
        )

    def _setup_axes(self) -> None:
        ax = self._ax
        ax.set_facecolor("#181825")
        ax.set_title("Live Frequency Spectrum", color="#cdd6f4", pad=8)
        ax.set_xlabel("Frequency (Hz)", color="#cdd6f4")
        ax.set_ylabel("Magnitude", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.set_xlim(0, MAX_FREQ_HZ)
        ax.set_ylim(0, 1)
        ax.grid(color="#313244", linestyle="--", linewidth=0.5)

    # ── Recording control ──────────────────────────────────────────────────────
    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FFT_CHUNK // 4,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._recording = True
            self._btn_toggle.config(text="⏹  Stop Recording")
            self._btn_capture.config(state=tk.NORMAL)
            self._status_var.set("Recording…")
            self._schedule_plot_update()
        except Exception as exc:
            messagebox.showerror("Audio Error", str(exc))

    def _stop_recording(self) -> None:
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._btn_toggle.config(text="▶  Start Recording")
        self._btn_capture.config(state=tk.DISABLED)
        self._status_var.set("Idle")
        self._peak_var.set("–")

    # ── Audio callback (runs in a background thread) ───────────────────────────
    def _audio_callback(
        self, indata: np.ndarray, frames: int, time, status
    ) -> None:
        with self._lock:
            chunk = indata[:, 0]
            n = len(chunk)
            self._audio_buffer = np.roll(self._audio_buffer, -n)
            self._audio_buffer[-n:] = chunk

    # ── Plot update (runs on the GUI thread via after()) ───────────────────────
    def _schedule_plot_update(self) -> None:
        if not self._recording:
            return
        self._update_plot()
        self.after(PLOT_INTERVAL_MS, self._schedule_plot_update)

    def _update_plot(self) -> None:
        with self._lock:
            data = self._audio_buffer.copy()

        window   = np.hanning(len(data))
        spectrum = np.abs(np.fft.rfft(data * window))
        freqs    = np.fft.rfftfreq(len(data), d=1.0 / SAMPLE_RATE)

        # Keep only frequencies up to MAX_FREQ_HZ
        mask     = freqs <= MAX_FREQ_HZ
        freqs    = freqs[mask]
        spectrum = spectrum[mask]

        # Store for later capture
        with self._lock:
            self._last_freqs = freqs.copy()
            self._last_spectrum = spectrum.copy()

        # Normalise
        peak_magnitude = spectrum.max() or 1.0
        normalised     = spectrum / peak_magnitude

        self._line.set_xdata(freqs)
        self._line.set_ydata(normalised)
        self._ax.set_xlim(0, MAX_FREQ_HZ)

        # Report peak frequency (ignore DC component)
        if len(spectrum) > 1:
            peak_idx  = np.argmax(spectrum[1:]) + 1
            peak_freq = freqs[peak_idx]
            self._peak_var.set(f"{peak_freq:.1f} Hz")

        self._canvas.draw_idle()

    def _capture_to_synthesizer(self) -> None:
        """Extract top-n frequencies and send to synthesizer."""
        if self._synthesizer_frame is None:
            messagebox.showwarning("Synthesizer Not Ready", "Synthesizer frame not initialized.")
            return

        with self._lock:
            freqs = self._last_freqs.copy()
            spectrum = self._last_spectrum.copy()

        if len(spectrum) == 0:
            messagebox.showwarning("No Data", "No frequency data available. Ensure recording is active.")
            return

        # Get top-n frequencies
        try:
            n = int(self._capture_count_var.get())
        except (tk.TclError, ValueError):
            n = 5

        n = min(n, len(spectrum))

        # Find top-n peaks (ignore DC)
        top_indices = np.argsort(spectrum[1:])[-n:] + 1
        top_indices = sorted(top_indices, reverse=True)
        top_frequencies = freqs[top_indices]
        top_magnitudes = spectrum[top_indices]

        # Normalize magnitudes to [0, 1] range
        peak_magnitude = spectrum.max() or 1.0
        top_amplitudes = top_magnitudes / peak_magnitude

        # Add to synthesizer with magnitudes
        self._synthesizer_frame.add_frequencies(top_frequencies, top_amplitudes)
        messagebox.showinfo("Success", f"Added {len(top_frequencies)} frequencies to synthesizer.")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def destroy(self) -> None:
        if self._recording:
            self._stop_recording()
        super().destroy()


class TunerFrame(ttk.Frame):
    """
    Tab 3 – Real-time pitch tuner with note detection and deviation visualization.
    Uses A=440Hz standard and displays deviation in cents from the nearest note.
    """

    A4_FREQ = 440.0  # Hz (concert pitch)
    SEMITONE_RATIO = 2.0 ** (1.0 / 12.0)  # ratio between adjacent semitones
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    IN_TUNE_THRESHOLD_CENTS = 3.0
    MIN_DB_THRESHOLD = -35.0  # dB (catches piano voice and soft speech)
    SMOOTHING_FRAMES = 8  # frames for frequency moving average

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._audio_buffer = np.zeros(FFT_CHUNK, dtype=np.float32)
        self._meter_info = {}
        self._freq_history = []  # for smoothing frequency readings
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        control_bar = ttk.Frame(self)
        control_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self._btn_toggle = ttk.Button(
            control_bar, text="▶  Start Tuning", command=self._toggle_recording
        )
        self._btn_toggle.pack(side=tk.LEFT, padx=4)

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(control_bar, textvariable=self._status_var).pack(side=tk.LEFT, padx=12)

        # Main display area
        display_frame = ttk.Frame(self)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Detected frequency
        ttk.Label(display_frame, text="Detected Frequency:", font=("Helvetica", 10)).pack(pady=4)
        self._freq_var = tk.StringVar(value="–")
        ttk.Label(
            display_frame, textvariable=self._freq_var, font=("Helvetica", 14, "bold")
        ).pack()

        # Nearest note
        ttk.Label(display_frame, text="Nearest Note:", font=("Helvetica", 10)).pack(pady=(12, 4))
        self._note_var = tk.StringVar(value="–")
        ttk.Label(display_frame, textvariable=self._note_var, font=("Helvetica", 20, "bold")).pack()

        # Deviation meter
        ttk.Label(display_frame, text="Tuning Status:", font=("Helvetica", 10)).pack(pady=(12, 4))
        self._meter_canvas = tk.Canvas(
            display_frame, width=300, height=70, bg="#181825", highlightthickness=0
        )
        self._meter_canvas.pack()
        self._draw_meter_background()

        # Deviation in cents
        self._deviation_var = tk.StringVar(value="– cents")
        ttk.Label(
            display_frame, textvariable=self._deviation_var, font=("Helvetica", 11, "bold")
        ).pack(pady=8)

    def _draw_meter_background(self) -> None:
        """Draw the baseline tuning meter."""
        canvas = self._meter_canvas
        canvas.delete("all")

        width, height = 300, 70
        center_x = width / 2
        center_y = height / 2.5

        # Background bar
        bar_width = 250
        bar_height = 20
        canvas.create_rectangle(
            center_x - bar_width / 2,
            center_y - bar_height / 2,
            center_x + bar_width / 2,
            center_y + bar_height / 2,
            fill="#313244",
            outline="#45475a",
        )

        # Flat zone (left)
        canvas.create_rectangle(
            center_x - bar_width / 2,
            center_y - bar_height / 2,
            center_x - bar_width / 3,
            center_y + bar_height / 2,
            fill="#3a1a1a",
            outline="",
        )

        # Sharp zone (right)
        canvas.create_rectangle(
            center_x + bar_width / 3,
            center_y - bar_height / 2,
            center_x + bar_width / 2,
            center_y + bar_height / 2,
            fill="#1a2a1a",
            outline="",
        )

        # Center line (perfect pitch)
        canvas.create_line(center_x, center_y - bar_height / 2 - 8, center_x, center_y + bar_height / 2 + 8, fill="#a6e3a1", width=3)

        self._meter_info = {
            "center_x": center_x,
            "center_y": center_y,
            "bar_width": bar_width,
            "bar_height": bar_height,
        }

    # ── Recording control ──────────────────────────────────────────────────────
    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        try:
            self._freq_history.clear()  # Reset smoothing buffer
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FFT_CHUNK // 4,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._recording = True
            self._btn_toggle.config(text="⏹  Stop Tuning")
            self._status_var.set("Tuning…")
            self._schedule_update()
        except Exception as exc:
            messagebox.showerror("Audio Error", str(exc))

    def _stop_recording(self) -> None:
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._btn_toggle.config(text="▶  Start Tuning")
        self._status_var.set("Idle")
        self._freq_var.set("–")
        self._note_var.set("–")
        self._deviation_var.set("– cents")

    # ── Audio callback ────────────────────────────────────────────────────────
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            chunk = indata[:, 0]
            n = len(chunk)
            self._audio_buffer = np.roll(self._audio_buffer, -n)
            self._audio_buffer[-n:] = chunk

    # ── Update loop ───────────────────────────────────────────────────────────
    def _schedule_update(self) -> None:
        if not self._recording:
            return
        self._update_tuner()
        self.after(PLOT_INTERVAL_MS, self._schedule_update)

    def _detect_frequency(self) -> float | None:
        """Detect the fundamental frequency using FFT peak detection."""
        with self._lock:
            data = self._audio_buffer.copy()

        # Check signal level (dB)
        rms = np.sqrt(np.mean(data ** 2))
        db_level = 20.0 * np.log10(rms) if rms > 1e-10 else -np.inf

        # Require minimum signal level to filter out background noise
        if db_level < self.MIN_DB_THRESHOLD:
            return None

        window = np.hanning(len(data))
        spectrum = np.abs(np.fft.rfft(data * window))
        freqs = np.fft.rfftfreq(len(data), d=1.0 / SAMPLE_RATE)

        # Focus on range suitable for tuning (50 Hz – 2000 Hz)
        mask = (freqs >= 50) & (freqs <= 2000)
        freqs = freqs[mask]
        spectrum = spectrum[mask]

        if len(spectrum) > 0:
            peak_idx = np.argmax(spectrum)
            detected_freq = float(freqs[peak_idx])

            # Apply smoothing via moving average
            self._freq_history.append(detected_freq)
            if len(self._freq_history) > self.SMOOTHING_FRAMES:
                self._freq_history.pop(0)

            smoothed_freq = np.mean(self._freq_history)
            return smoothed_freq
        return None

    def _get_nearest_note(self, frequency: float) -> tuple[str, float, float]:
        """
        Find the nearest musical note to the given frequency.
        Returns (note_name, note_frequency, cents_deviation)
        """
        # Generate frequencies for all notes from C0 onwards
        c0_freq = 16.35  # C0 frequency
        all_notes = []

        for octave in range(0, 9):
            for semitone in range(12):
                freq = c0_freq * (self.SEMITONE_RATIO ** (octave * 12 + semitone))
                note_name = f"{self.NOTE_NAMES[semitone]}{octave}"
                all_notes.append((note_name, freq))

        # Find the closest note
        nearest_note = min(all_notes, key=lambda x: abs(x[1] - frequency))
        note_name, note_freq = nearest_note

        # Calculate deviation in cents (1 semitone = 100 cents)
        cents_dev = 1200.0 * np.log2(frequency / note_freq)

        return note_name, note_freq, cents_dev

    def _update_tuner(self) -> None:
        """Update the tuner display with detected pitch and deviation."""
        detected_freq = self._detect_frequency()

        if detected_freq is None or detected_freq < 50:
            self._freq_var.set("–")
            self._note_var.set("–")
            self._deviation_var.set("– cents")
            return

        self._freq_var.set(f"{detected_freq:.1f} Hz")

        note_name, note_freq, cents_dev = self._get_nearest_note(detected_freq)
        self._note_var.set(note_name)

        deviation_str = f"{cents_dev:+.1f} cents"
        self._deviation_var.set(deviation_str)

        self._update_meter(cents_dev)

    def _update_meter(self, cents_deviation: float) -> None:
        """Update the tuning needle and colors based on deviation."""
        self._draw_meter_background()
        canvas = self._meter_canvas
        info = self._meter_info

        center_x = info["center_x"]
        center_y = info["center_y"]
        bar_width = info["bar_width"]
        bar_height = info["bar_height"]

        # Constrain display to ±50 cents
        max_display_cents = 50.0
        clamped_deviation = max(-max_display_cents, min(max_display_cents, cents_deviation))
        needle_offset = (clamped_deviation / max_display_cents) * (bar_width / 2)
        needle_x = center_x + needle_offset

        # Color based on tuning status
        if abs(cents_deviation) < self.IN_TUNE_THRESHOLD_CENTS:
            color = "#a6e3a1"  # Green – in tune
        elif cents_deviation > 0:
            color = "#f38ba8"  # Red/Pink – sharp (too high)
        else:
            color = "#fab387"  # Orange/Red – flat (too low)

        # Draw the needle
        canvas.create_line(
            needle_x, center_y - bar_height / 2 - 12, needle_x, center_y + bar_height / 2 + 12, fill=color, width=4
        )

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def destroy(self) -> None:
        if self._recording:
            self._stop_recording()
        super().destroy()


class SynthComponent(ttk.Frame):
    """
    A single row in the synthesizer representing one sine-wave component.
    """

    def __init__(self, parent: tk.Widget, index: int, on_remove, on_more=None):
        super().__init__(parent)
        self._index    = index
        self._on_remove = on_remove
        self._on_more = on_more
        self._build_ui()

    def _build_ui(self) -> None:
        ttk.Label(self, text=f"Wave {self._index}").grid(row=0, column=0, padx=4, pady=2)

        # Frequency
        ttk.Label(self, text="Freq (Hz):").grid(row=0, column=1, pady=2)
        self._freq_var = tk.DoubleVar(value=440.0)
        freq_spin = tk.Spinbox(
            self, from_=20, to=20000, increment=1, textvariable=self._freq_var, width=7,
            background="#313244", foreground="#cdd6f4", insertbackground="#a6e3a1",
            insertwidth=2, relief="sunken", borderwidth=1
        )
        freq_spin.grid(row=0, column=2, padx=4, pady=2)
        freq_spin.bind("<FocusIn>", lambda e: freq_spin.select_range(0, tk.END))

        # Amplitude
        ttk.Label(self, text="Amplitude:").grid(row=0, column=3, pady=2)
        self._amp_var = tk.DoubleVar(value=1.0)
        amp_spin = ttk.Spinbox(
            self, from_=0.0, to=1.0, increment=0.05, textvariable=self._amp_var,
            width=6, format="%.2f"
        )
        amp_spin.grid(row=0, column=4, padx=4, pady=2)

        # Phase offset (degrees)
        ttk.Label(self, text="Phase (°):").grid(row=0, column=5, pady=2)
        self._phase_var = tk.DoubleVar(value=0.0)
        phase_spin = ttk.Spinbox(
            self, from_=0, to=360, increment=1, textvariable=self._phase_var, width=6
        )
        phase_spin.grid(row=0, column=6, padx=4, pady=2)

        # More options button
        more_btn = ttk.Button(self, text="⋯", width=2, command=self._show_more_menu)
        more_btn.grid(row=0, column=7, padx=1, pady=2)

        # Octave up/down buttons
        ttk.Button(self, text="▼", width=2, command=self._octave_down).grid(
            row=0, column=8, padx=1, pady=2
        )
        ttk.Button(self, text="▲", width=2, command=self._octave_up).grid(
            row=0, column=9, padx=1, pady=2
        )

        ttk.Button(self, text="✕", width=3, command=self._on_remove).grid(
            row=0, column=10, padx=4, pady=2
        )

    @property
    def frequency(self) -> float:
        try:
            return float(self._freq_var.get())
        except tk.TclError:
            return 440.0

    @property
    def amplitude(self) -> float:
        try:
            return max(0.0, min(1.0, float(self._amp_var.get())))
        except tk.TclError:
            return 1.0

    @property
    def phase_rad(self) -> float:
        try:
            return float(self._phase_var.get()) * np.pi / 180.0
        except tk.TclError:
            return 0.0

    def _octave_up(self) -> None:
        """Multiply frequency by 2 (raise octave)."""
        try:
            current_freq = float(self._freq_var.get())
            new_freq = min(20000, current_freq * 2.0)  # Cap at 20kHz
            self._freq_var.set(new_freq)
        except tk.TclError:
            pass

    def _octave_down(self) -> None:
        """Divide frequency by 2 (lower octave)."""
        try:
            current_freq = float(self._freq_var.get())
            new_freq = max(20, current_freq / 2.0)  # Floor at 20Hz
            self._freq_var.set(new_freq)
        except tk.TclError:
            pass

    def _show_more_menu(self) -> None:
        """Show context menu with additional options."""
        if self._on_more is None:
            return

        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Add Harmonics", command=lambda: self._on_more("harmonics", self._index))
        menu.post(self.winfo_pointerx(), self.winfo_pointery())


class SynthesizerFrame(ttk.Frame):
    """
    Tab 2 – Build a waveform from sine-wave components and play it.
    """

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._components: list[SynthComponent] = []
        self._component_counter = 0
        self._playback_active = False
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Top controls
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(ctrl, text="➕  Add Sine Wave", command=self._add_component).pack(
            side=tk.LEFT, padx=4
        )
        self._btn_play = ttk.Button(
            ctrl, text="▶  Play", command=self._toggle_playback
        )
        self._btn_play.pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl, text="🗑  Clear All", command=self._clear_all_components).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Button(ctrl, text="📊  Spectrogram", command=self._generate_spectrogram).pack(
            side=tk.LEFT, padx=4
        )

        # Sort buttons
        ttk.Label(ctrl, text="  Sort by:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(ctrl, text="Freq", width=5, command=lambda: self._sort_components("frequency")).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(ctrl, text="Mag", width=5, command=lambda: self._sort_components("amplitude")).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(ctrl, text="Order", width=5, command=lambda: self._sort_components("order")).pack(
            side=tk.LEFT, padx=1
        )

        # Duration
        ttk.Label(ctrl, text="Duration (s):").pack(side=tk.LEFT, padx=(12, 2))
        self._duration_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(
            ctrl, from_=0.5, to=30.0, increment=0.5,
            textvariable=self._duration_var, width=5, format="%.1f"
        ).pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(ctrl, textvariable=self._status_var).pack(side=tk.LEFT, padx=12)

        # Scrollable list of sine-wave components
        list_outer = ttk.LabelFrame(self, text="Sine-Wave Components")
        list_outer.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

        canvas_scroll = tk.Canvas(list_outer, height=220)
        scrollbar = ttk.Scrollbar(list_outer, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self._list_inner = ttk.Frame(canvas_scroll)
        self._list_inner.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        canvas_scroll.create_window((0, 0), window=self._list_inner, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store canvas_scroll reference for scrollbar updates
        self._canvas_scroll = canvas_scroll

        # Bind mouse wheel scrolling to canvas
        canvas_scroll.bind("<MouseWheel>", self._on_mousewheel)
        canvas_scroll.bind("<Button-4>", self._on_mousewheel)  # Linux scroll up
        canvas_scroll.bind("<Button-5>", self._on_mousewheel)  # Linux scroll down

        # Preset notes (standard western 12 notes)
        notes_frame = ttk.LabelFrame(self, text="Quick Add: Standard Notes (A=440Hz)")
        notes_frame.pack(fill=tk.X, padx=8, pady=4)
        self._note_freqs = {
            "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
            "E": 329.63, "F": 349.23, "F#": 369.99, "G": 391.99,
            "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
        }
        notes_inner = ttk.Frame(notes_frame)
        notes_inner.pack(fill=tk.X, padx=4, pady=4)
        for note_name, freq in self._note_freqs.items():
            ttk.Button(
                notes_inner, text=note_name, width=4,
                command=lambda f=freq: self._add_frequency(f)
            ).pack(side=tk.LEFT, padx=1)

        # Preview plot
        plot_frame = ttk.LabelFrame(self, text="Waveform Preview")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        fig = Figure(figsize=(9, 2.5), dpi=96, facecolor="#1e1e2e")
        self._ax = fig.add_subplot(111)
        self._setup_axes()
        (self._line,) = self._ax.plot([], [], color="#a6e3a1", lw=1.2)

        self._canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Seed with one default component
        self._add_component()

    def _setup_axes(self) -> None:
        ax = self._ax
        ax.set_facecolor("#181825")
        ax.set_title("Combined Waveform (first 50 ms)", color="#cdd6f4", pad=6)
        ax.set_xlabel("Time (ms)", color="#cdd6f4")
        ax.set_ylabel("Amplitude", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.grid(color="#313244", linestyle="--", linewidth=0.5)

    # ── Component management ───────────────────────────────────────────────────
    def _add_component(self) -> None:
        self._component_counter += 1
        idx  = self._component_counter
        comp = SynthComponent(
            self._list_inner, idx,
            on_remove=lambda i=idx: self._remove_component(i),
            on_more=self._handle_component_option
        )
        comp.pack(fill=tk.X, pady=2, padx=4)
        self._components.append(comp)
        self._refresh_preview()

    def _remove_component(self, index: int) -> None:
        to_remove = [c for c in self._components if c._index == index]
        for comp in to_remove:
            self._components.remove(comp)
            comp.destroy()
        self._refresh_preview()

    def add_frequencies(self, frequencies: np.ndarray, amplitudes: np.ndarray | None = None) -> None:
        """Add multiple frequency components from the analyzer."""
        if amplitudes is None:
            amplitudes = np.ones_like(frequencies)

        for freq, amp in zip(frequencies, amplitudes):
            if freq > 0:
                self._component_counter += 1
                idx = self._component_counter
                comp = SynthComponent(
                    self._list_inner, idx,
                    on_remove=lambda i=idx: self._remove_component(i),
                    on_more=self._handle_component_option
                )
                comp.pack(fill=tk.X, pady=2, padx=4)
                self._components.append(comp)
                # Set frequency and amplitude
                comp._freq_var.set(float(freq))
                comp._amp_var.set(float(amp))
        self._refresh_preview()

    def _clear_all_components(self) -> None:
        """Remove all sine-wave components."""
        for comp in list(self._components):
            self._components.remove(comp)
            comp.destroy()
        # Refresh scrollbar to reflect empty list
        self._canvas_scroll.configure(scrollregion=self._canvas_scroll.bbox("all"))
        self._refresh_preview()

    def _add_frequency(self, frequency: float) -> None:
        """Add a single frequency component."""
        self.add_frequencies(np.array([frequency]), np.array([1.0]))

    def _handle_component_option(self, option: str, comp_index: int) -> None:
        """Handle component menu options."""
        if option == "harmonics":
            self._add_harmonics_dialog(comp_index)

    def _add_harmonics_dialog(self, comp_index: int) -> None:
        """Ask user how many harmonics to add and generate them."""
        # Find the component
        comp = next((c for c in self._components if c._index == comp_index), None)
        if comp is None:
            return

        # Create a simple dialog
        dialog = tk.Toplevel(self)
        dialog.title("Add Harmonics")
        dialog.geometry("300x150")
        dialog.configure(bg="#1e1e2e")

        ttk.Label(dialog, text="Number of harmonics to add:").pack(pady=10)
        harmonics_var = tk.IntVar(value=3)
        ttk.Spinbox(dialog, from_=1, to=10, textvariable=harmonics_var, width=10).pack()

        def add_harmonics():
            try:
                n_harmonics = int(harmonics_var.get())
                fundamental_freq = comp.frequency
                fundamental_amp = comp.amplitude

                harmonics_freqs = []
                harmonics_amps = []

                for harmonic_num in range(2, n_harmonics + 2):
                    harmonic_freq = fundamental_freq * harmonic_num
                    # Scale amplitude as 1/n^2 (realistic harmonic decay)
                    harmonic_amp = fundamental_amp / (harmonic_num ** 2)
                    harmonics_freqs.append(harmonic_freq)
                    harmonics_amps.append(harmonic_amp)

                # Add harmonics
                if harmonics_freqs:
                    self.add_frequencies(np.array(harmonics_freqs), np.array(harmonics_amps))

                dialog.destroy()
                messagebox.showinfo("Success", f"Added {len(harmonics_freqs)} harmonics")
            except (ValueError, tk.TclError):
                messagebox.showerror("Error", "Invalid input")

        ttk.Button(dialog, text="Add", command=add_harmonics).pack(pady=10)

    def _sort_components(self, sort_by: str) -> None:
        """Sort components by frequency, amplitude, or order added."""
        if sort_by == "frequency":
            self._components.sort(key=lambda c: c.frequency)
        elif sort_by == "amplitude":
            self._components.sort(key=lambda c: c.amplitude, reverse=True)
        elif sort_by == "order":
            self._components.sort(key=lambda c: c._index)

        # Clear and re-pack components in new order
        for comp in self._components:
            comp.pack_forget()  # Unpack first

        for comp in self._components:
            comp.pack(fill=tk.X, pady=2, padx=4)

        # Force layout update and scroll region refresh
        self._list_inner.update_idletasks()
        self._canvas_scroll.update_idletasks()
        self._canvas_scroll.configure(scrollregion=self._canvas_scroll.bbox("all"))
        self._refresh_preview()

    def _generate_spectrogram(self) -> None:
        """Generate and display spectrogram from synthesizer frequencies."""
        if not self._components:
            messagebox.showwarning("No Components", "Add at least one frequency component first.")
            return

        try:
            duration = float(self._duration_var.get())
        except tk.TclError:
            duration = 3.0

        # Generate waveform
        signal = self._build_waveform(duration)
        n_samples = len(signal)

        # Compute spectrogram
        from scipy import signal as sp_signal
        freqs, times, Sxx = sp_signal.spectrogram(signal, fs=SAMPLE_RATE, nperseg=1024)

        # Create a new window with spectrogram
        spec_window = tk.Toplevel(self)
        spec_window.title("Spectrogram")
        spec_window.geometry("900x600")

        fig = Figure(figsize=(9, 5.5), dpi=96, facecolor="#1e1e2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#181825")

        # Plot spectrogram (log scale for better visualization)
        pcm = ax.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)', color="#cdd6f4")
        ax.set_xlabel('Time (s)', color="#cdd6f4")
        ax.set_title('Spectrogram', color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.set_ylim(0, min(1000, SAMPLE_RATE / 2))  # Limit y-axis for better visibility

        fig.colorbar(pcm, ax=ax, label="Power (dB)")

        canvas = FigureCanvasTkAgg(fig, master=spec_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    # ── Waveform generation ────────────────────────────────────────────────────
    def _build_waveform(self, duration: float) -> np.ndarray:
        n_samples = int(SAMPLE_RATE * duration)
        t         = np.linspace(0, duration, n_samples, endpoint=False)
        signal    = np.zeros(n_samples, dtype=np.float64)
        for comp in self._components:
            signal += comp.amplitude * np.sin(
                2.0 * np.pi * comp.frequency * t + comp.phase_rad
            )
        # Normalise to [-1, 1] to avoid clipping
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal /= peak
        return signal.astype(np.float32)

    def _refresh_preview(self) -> None:
        preview_duration = 0.05   # 50 ms
        signal = self._build_waveform(preview_duration)
        t_ms   = np.linspace(0, preview_duration * 1000, len(signal), endpoint=False)
        self._line.set_xdata(t_ms)
        self._line.set_ydata(signal)
        self._ax.set_xlim(0, preview_duration * 1000)
        self._ax.set_ylim(-1.1, 1.1)
        self._canvas.draw_idle()

    def _on_mousewheel(self, event) -> None:
        """Handle mouse wheel scrolling on the components canvas."""
        # Determine scroll direction
        if event.num == 5 or event.delta < 0:
            self._canvas_scroll.yview_scroll(3, "units")
        else:  # event.num == 4 or event.delta > 0
            self._canvas_scroll.yview_scroll(-3, "units")

    # ──  Playback ───────────────────────────────────────────────────────────────
    def _toggle_playback(self) -> None:
        if self._playback_active:
            sd.stop()
            self._playback_active = False
            self._btn_play.config(text="▶  Play")
            self._status_var.set("Stopped")
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if not self._components:
            messagebox.showinfo("No Components", "Add at least one sine-wave component first.")
            return
        try:
            duration = float(self._duration_var.get())
        except tk.TclError:
            duration = 3.0
        self._refresh_preview()
        signal = self._build_waveform(duration)
        self._playback_active = True
        self._btn_play.config(text="⏹  Stop")
        self._status_var.set("Playing…")

        def _done_callback():
            self._playback_active = False
            self._btn_play.config(text="▶  Play")
            self._status_var.set("Done")

        def _play_thread():
            sd.play(signal, samplerate=SAMPLE_RATE)
            sd.wait()
            self.after(0, _done_callback)

        threading.Thread(target=_play_thread, daemon=True).start()


class AcousticsApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Acoustics Lab")
        self.minsize(800, 560)
        self.configure(bg="#1e1e2e")

        self._apply_theme()
        self._build_ui()

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        bg, fg, accent = "#1e1e2e", "#cdd6f4", "#89b4fa"
        style.configure(".", background=bg, foreground=fg, fieldbackground=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background="#313244", foreground=fg, relief="flat")
        style.map("TButton", background=[("active", accent), ("pressed", "#74c7ec")])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground=fg, padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", accent)], foreground=[("selected", bg)])
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg, foreground=accent)
        style.configure("TSpinbox", background="#313244", foreground=fg, fieldbackground="#313244")
        style.configure("Vertical.TScrollbar", background="#313244")

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._analyzer   = AnalyzerFrame(notebook)
        self._tuner      = TunerFrame(notebook)
        self._synthesizer = SynthesizerFrame(notebook)

        # Wire analyzer to synthesizer for frequency capture
        self._analyzer._synthesizer_frame = self._synthesizer

        notebook.add(self._analyzer,    text="🎤  Analyzer")
        notebook.add(self._tuner,       text="🎵  Tuner")
        notebook.add(self._synthesizer, text="🎹  Synthesizer")

        # Stop recording when switching frames
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event) -> None:
        """Stop all recordings when switching tabs."""
        if self._analyzer._recording:
            self._analyzer._stop_recording()
        if self._tuner._recording:
            self._tuner._stop_recording()

    def destroy(self) -> None:
        if hasattr(self, "_analyzer"):
            self._analyzer.destroy()
        if hasattr(self, "_tuner"):
            self._tuner.destroy()
        super().destroy()


def main() -> None:
    app = AcousticsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
