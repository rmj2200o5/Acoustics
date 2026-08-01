"""Tuner tab UI and pitch detection logic."""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd

from .constants import FFT_CHUNK, PLOT_INTERVAL_MS, SAMPLE_RATE


class TunerFrame(ttk.Frame):
    """
    Tab 3 – Real-time pitch tuner with note detection and deviation visualization.
    Uses A=440Hz standard and displays deviation in cents from the nearest note.
    """

    A4_FREQ = 440.0
    SEMITONE_RATIO = 2.0 ** (1.0 / 12.0)
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    IN_TUNE_THRESHOLD_CENTS = 3.0
    MIN_DB_THRESHOLD = -35.0
    SMOOTHING_FRAMES = 8

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._audio_buffer = np.zeros(FFT_CHUNK, dtype=np.float32)
        self._meter_info = {}
        self._freq_history = []
        self._build_ui()

    def _build_ui(self) -> None:
        control_bar = ttk.Frame(self)
        control_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self._btn_toggle = ttk.Button(
            control_bar, text="▶  Start Tuning", command=self._toggle_recording
        )
        self._btn_toggle.pack(side=tk.LEFT, padx=4)

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(control_bar, textvariable=self._status_var).pack(side=tk.LEFT, padx=12)

        display_frame = ttk.Frame(self)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(display_frame, text="Detected Frequency:", font=("Helvetica", 10)).pack(pady=4)
        self._freq_var = tk.StringVar(value="–")
        ttk.Label(
            display_frame, textvariable=self._freq_var, font=("Helvetica", 14, "bold")
        ).pack()

        ttk.Label(display_frame, text="Nearest Note:", font=("Helvetica", 10)).pack(pady=(12, 4))
        self._note_var = tk.StringVar(value="–")
        ttk.Label(display_frame, textvariable=self._note_var, font=("Helvetica", 20, "bold")).pack()

        ttk.Label(display_frame, text="Tuning Status:", font=("Helvetica", 10)).pack(pady=(12, 4))
        self._meter_canvas = tk.Canvas(
            display_frame, width=300, height=70, bg="#181825", highlightthickness=0
        )
        self._meter_canvas.pack()
        self._draw_meter_background()

        self._deviation_var = tk.StringVar(value="– cents")
        ttk.Label(
            display_frame, textvariable=self._deviation_var, font=("Helvetica", 11, "bold")
        ).pack(pady=8)

    def _draw_meter_background(self) -> None:
        canvas = self._meter_canvas
        canvas.delete("all")

        width, height = 300, 70
        center_x = width / 2
        center_y = height / 2.5

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

        canvas.create_rectangle(
            center_x - bar_width / 2,
            center_y - bar_height / 2,
            center_x - bar_width / 3,
            center_y + bar_height / 2,
            fill="#3a1a1a",
            outline="",
        )

        canvas.create_rectangle(
            center_x + bar_width / 3,
            center_y - bar_height / 2,
            center_x + bar_width / 2,
            center_y + bar_height / 2,
            fill="#1a2a1a",
            outline="",
        )

        canvas.create_line(
            center_x,
            center_y - bar_height / 2 - 8,
            center_x,
            center_y + bar_height / 2 + 8,
            fill="#a6e3a1",
            width=3,
        )

        self._meter_info = {
            "center_x": center_x,
            "center_y": center_y,
            "bar_width": bar_width,
            "bar_height": bar_height,
        }

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        try:
            self._freq_history.clear()
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

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            chunk = indata[:, 0]
            n = len(chunk)
            self._audio_buffer = np.roll(self._audio_buffer, -n)
            self._audio_buffer[-n:] = chunk

    def _schedule_update(self) -> None:
        if not self._recording:
            return
        self._update_tuner()
        self.after(PLOT_INTERVAL_MS, self._schedule_update)

    def _detect_frequency(self) -> float | None:
        with self._lock:
            data = self._audio_buffer.copy()

        rms = np.sqrt(np.mean(data**2))
        db_level = 20.0 * np.log10(rms) if rms > 1e-10 else -np.inf
        if db_level < self.MIN_DB_THRESHOLD:
            return None

        window = np.hanning(len(data))
        spectrum = np.abs(np.fft.rfft(data * window))
        freqs = np.fft.rfftfreq(len(data), d=1.0 / SAMPLE_RATE)

        mask = (freqs >= 50) & (freqs <= 2000)
        freqs = freqs[mask]
        spectrum = spectrum[mask]

        if len(spectrum) > 0:
            peak_idx = np.argmax(spectrum)
            detected_freq = float(freqs[peak_idx])

            self._freq_history.append(detected_freq)
            if len(self._freq_history) > self.SMOOTHING_FRAMES:
                self._freq_history.pop(0)

            smoothed_freq = np.mean(self._freq_history)
            return smoothed_freq
        return None

    def _get_nearest_note(self, frequency: float) -> tuple[str, float, float]:
        c0_freq = 16.35
        all_notes = []

        for octave in range(0, 9):
            for semitone in range(12):
                freq = c0_freq * (self.SEMITONE_RATIO ** (octave * 12 + semitone))
                note_name = f"{self.NOTE_NAMES[semitone]}{octave}"
                all_notes.append((note_name, freq))

        nearest_note = min(all_notes, key=lambda x: abs(x[1] - frequency))
        note_name, note_freq = nearest_note
        cents_dev = 1200.0 * np.log2(frequency / note_freq)

        return note_name, note_freq, cents_dev

    def _update_tuner(self) -> None:
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
        self._draw_meter_background()
        canvas = self._meter_canvas
        info = self._meter_info

        center_x = info["center_x"]
        center_y = info["center_y"]
        bar_width = info["bar_width"]
        bar_height = info["bar_height"]

        max_display_cents = 50.0
        clamped_deviation = max(-max_display_cents, min(max_display_cents, cents_deviation))
        needle_offset = (clamped_deviation / max_display_cents) * (bar_width / 2)
        needle_x = center_x + needle_offset

        if abs(cents_deviation) < self.IN_TUNE_THRESHOLD_CENTS:
            color = "#a6e3a1"
        elif cents_deviation > 0:
            color = "#f38ba8"
        else:
            color = "#fab387"

        canvas.create_line(
            needle_x,
            center_y - bar_height / 2 - 12,
            needle_x,
            center_y + bar_height / 2 + 12,
            fill=color,
            width=4,
        )

    def destroy(self) -> None:
        if self._recording:
            self._stop_recording()
        super().destroy()
