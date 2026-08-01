"""Analyzer tab UI and real-time FFT/spectrogram logic."""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .constants import FFT_CHUNK, MAX_FREQ_HZ, PLOT_INTERVAL_MS, SAMPLE_RATE


class AnalyzerFrame(ttk.Frame):
    """
    Tab 1 - Microphone → FFT → Live analyzer view (Spectrum / Spectrogram).
    """

    MIN_DB_THRESHOLD = -30.0
    SPEC_HISTORY_FRAMES = 180
    SPECTROGRAM_FLOOR_DB = -100.0

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._audio_buffer = np.zeros(FFT_CHUNK, dtype=np.float32)
        self._last_freqs = np.array([])
        self._last_spectrum = np.array([])
        self._synthesizer_frame = None  # Will be set by parent app

        self._view_mode_var = tk.StringVar(value="Spectrum")

        # Precompute FFT frequency axis/mask for plotting
        self._fft_freqs = np.fft.rfftfreq(FFT_CHUNK, d=1.0 / SAMPLE_RATE)
        self._fft_mask = self._fft_freqs <= MAX_FREQ_HZ
        self._plot_freqs = self._fft_freqs[self._fft_mask]
        self._n_bins = len(self._plot_freqs)

        self._build_ui()

    def _build_ui(self) -> None:
        control_bar = ttk.Frame(self)
        control_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self._btn_toggle = ttk.Button(
            control_bar, text="▶  Start Recording", command=self._toggle_recording
        )
        self._btn_toggle.pack(side=tk.LEFT, padx=4)

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(control_bar, textvariable=self._status_var).pack(side=tk.LEFT, padx=12)

        ttk.Separator(control_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(control_bar, text="Capture top").pack(side=tk.LEFT, padx=(4, 2))
        self._capture_count_var = tk.IntVar(value=5)
        ttk.Spinbox(
            control_bar, from_=1, to=20, textvariable=self._capture_count_var, width=4
        ).pack(side=tk.LEFT, padx=2)
        ttk.Label(control_bar, text="frequencies").pack(side=tk.LEFT, padx=2)

        self._btn_capture = ttk.Button(
            control_bar,
            text="→ Synthesizer",
            command=self._capture_to_synthesizer,
            state=tk.DISABLED,
        )
        self._btn_capture.pack(side=tk.LEFT, padx=4)

        ttk.Separator(control_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(control_bar, text="View:").pack(side=tk.LEFT, padx=(4, 2))
        self._view_combo = ttk.Combobox(
            control_bar,
            textvariable=self._view_mode_var,
            values=("Spectrum", "Spectrogram"),
            state="readonly",
            width=12,
        )
        self._view_combo.pack(side=tk.LEFT, padx=2)
        self._view_combo.bind("<<ComboboxSelected>>", self._on_view_mode_changed)

        fig = Figure(figsize=(9, 4), dpi=96, facecolor="#1e1e2e")
        self._ax = fig.add_subplot(111)
        self._setup_axes()

        (self._line,) = self._ax.plot(
            self._plot_freqs, np.zeros(self._n_bins), color="#89b4fa", lw=1.2
        )

        self._spec_data = np.full(
            (self._n_bins, self.SPEC_HISTORY_FRAMES),
            self.SPECTROGRAM_FLOOR_DB,
            dtype=np.float32,
        )
        time_span_s = (self.SPEC_HISTORY_FRAMES * PLOT_INTERVAL_MS) / 1000.0
        self._spec_im = self._ax.imshow(
            self._spec_data,
            origin="lower",
            aspect="auto",
            extent=(-time_span_s, 0, 0, MAX_FREQ_HZ),
            cmap="magma",
            vmin=-100,
            vmax=0,
            interpolation="nearest",
        )
        self._spec_im.set_visible(False)

        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        info_bar = ttk.Frame(self)
        info_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        ttk.Label(info_bar, text="Peak frequency:").pack(side=tk.LEFT)
        self._peak_var = tk.StringVar(value="–")
        ttk.Label(info_bar, textvariable=self._peak_var, font=("Helvetica", 12, "bold")).pack(
            side=tk.LEFT, padx=6
        )

        self._apply_view_mode()

    def _setup_axes(self) -> None:
        ax = self._ax
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.grid(color="#313244", linestyle="--", linewidth=0.5)

    def _on_view_mode_changed(self, _event=None) -> None:
        self._apply_view_mode()
        self._canvas.draw_idle()

    def _apply_view_mode(self) -> None:
        mode = self._view_mode_var.get()
        if mode == "Spectrogram":
            self._line.set_visible(False)
            self._spec_im.set_visible(True)
            self._ax.set_title("Live Spectrogram", color="#cdd6f4", pad=8)
            self._ax.set_xlabel("Time (s)", color="#cdd6f4")
            self._ax.set_ylabel("Frequency (Hz)", color="#cdd6f4")
            self._ax.set_xlim(self._spec_im.get_extent()[0], self._spec_im.get_extent()[1])
            self._ax.set_ylim(0, MAX_FREQ_HZ)
        else:
            self._line.set_visible(True)
            self._spec_im.set_visible(False)
            self._ax.set_title("Live Frequency Spectrum", color="#cdd6f4", pad=8)
            self._ax.set_xlabel("Frequency (Hz)", color="#cdd6f4")
            self._ax.set_ylabel("Magnitude", color="#cdd6f4")
            self._ax.set_xlim(0, MAX_FREQ_HZ)
            self._ax.set_ylim(0, 1)

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

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            chunk = indata[:, 0]
            n = len(chunk)
            self._audio_buffer = np.roll(self._audio_buffer, -n)
            self._audio_buffer[-n:] = chunk

    def _schedule_plot_update(self) -> None:
        if not self._recording:
            return
        self._update_plot()
        self.after(PLOT_INTERVAL_MS, self._schedule_plot_update)

    def _update_plot(self) -> None:
        with self._lock:
            data = self._audio_buffer.copy()

        rms = np.sqrt(np.mean(data**2))
        db_level = 20.0 * np.log10(rms) if rms > 1e-10 else -np.inf

        if db_level < self.MIN_DB_THRESHOLD:
            with self._lock:
                self._last_freqs = np.array([])
                self._last_spectrum = np.array([])

            if self._view_mode_var.get() == "Spectrum":
                self._line.set_xdata(self._plot_freqs)
                self._line.set_ydata(np.zeros_like(self._plot_freqs))
                self._ax.set_xlim(0, MAX_FREQ_HZ)
                self._ax.set_ylim(0, 1)
            else:
                self._spec_data = np.roll(self._spec_data, -1, axis=1)
                self._spec_data[:, -1] = self.SPECTROGRAM_FLOOR_DB
                self._spec_im.set_data(self._spec_data)
                self._ax.set_ylim(0, MAX_FREQ_HZ)

            self._peak_var.set("–")
            self._canvas.draw_idle()
            return

        window = np.hanning(len(data))
        spectrum_full = np.abs(np.fft.rfft(data * window))
        spectrum = spectrum_full[self._fft_mask]
        freqs = self._plot_freqs

        with self._lock:
            self._last_freqs = freqs.copy()
            self._last_spectrum = spectrum.copy()

        if len(spectrum) > 1:
            peak_idx = np.argmax(spectrum[1:]) + 1
            self._peak_var.set(f"{freqs[peak_idx]:.1f} Hz")
        else:
            self._peak_var.set("–")

        if self._view_mode_var.get() == "Spectrum":
            peak_magnitude = spectrum.max() or 1.0
            normalised = spectrum / peak_magnitude
            self._line.set_xdata(freqs)
            self._line.set_ydata(normalised)
            self._ax.set_xlim(0, MAX_FREQ_HZ)
            self._ax.set_ylim(0, 1)
        else:
            peak_magnitude = spectrum.max() or 1.0
            spec_db = 20.0 * np.log10((spectrum / peak_magnitude) + 1e-12)
            spec_db = np.clip(spec_db, self.SPECTROGRAM_FLOOR_DB, 0.0)

            self._spec_data = np.roll(self._spec_data, -1, axis=1)
            self._spec_data[:, -1] = spec_db
            self._spec_im.set_data(self._spec_data)
            self._ax.set_ylim(0, MAX_FREQ_HZ)

        self._canvas.draw_idle()

    def _capture_to_synthesizer(self) -> None:
        if self._synthesizer_frame is None:
            messagebox.showwarning("Synthesizer Not Ready", "Synthesizer frame not initialized.")
            return

        with self._lock:
            freqs = self._last_freqs.copy()
            spectrum = self._last_spectrum.copy()

        if len(spectrum) == 0:
            messagebox.showwarning("No Data", "No frequency data available. Ensure recording is active.")
            return

        try:
            n = int(self._capture_count_var.get())
        except (tk.TclError, ValueError):
            n = 5

        n = min(n, len(spectrum) - 1) if len(spectrum) > 1 else 0
        if n <= 0:
            messagebox.showwarning("No Data", "Not enough spectrum data to capture.")
            return

        top_indices = np.argsort(spectrum[1:])[-n:] + 1
        top_indices = top_indices[np.argsort(spectrum[top_indices])[::-1]]

        top_frequencies = freqs[top_indices]
        top_magnitudes = spectrum[top_indices]

        peak_magnitude = spectrum.max() or 1.0
        top_amplitudes = top_magnitudes / peak_magnitude

        self._synthesizer_frame.add_frequencies(top_frequencies, top_amplitudes)
        messagebox.showinfo("Success", f"Added {len(top_frequencies)} frequencies to synthesizer.")

    def destroy(self) -> None:
        if self._recording:
            self._stop_recording()
        super().destroy()
