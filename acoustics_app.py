"""
Acoustics Application
=====================
A GUI application with two main features:
  1. Analyzer – records audio from the microphone, performs a real-time FFT
     and displays a live frequency-spectrum plot.
  2. Synthesizer – lets the user build a waveform from one or more sine-wave
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
MAX_FREQ_HZ = 8000           # highest frequency shown on the spectrum plot
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

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def destroy(self) -> None:
        if self._recording:
            self._stop_recording()
        super().destroy()


class SynthComponent(ttk.Frame):
    """
    A single row in the synthesizer representing one sine-wave component.
    """

    def __init__(self, parent: tk.Widget, index: int, on_remove):
        super().__init__(parent)
        self._index    = index
        self._on_remove = on_remove
        self._build_ui()

    def _build_ui(self) -> None:
        ttk.Label(self, text=f"Wave {self._index}").grid(row=0, column=0, padx=4)

        # Frequency
        ttk.Label(self, text="Freq (Hz):").grid(row=0, column=1)
        self._freq_var = tk.DoubleVar(value=440.0)
        freq_spin = ttk.Spinbox(
            self, from_=20, to=20000, increment=1, textvariable=self._freq_var, width=7
        )
        freq_spin.grid(row=0, column=2, padx=4)

        # Amplitude
        ttk.Label(self, text="Amplitude:").grid(row=0, column=3)
        self._amp_var = tk.DoubleVar(value=1.0)
        amp_spin = ttk.Spinbox(
            self, from_=0.0, to=1.0, increment=0.05, textvariable=self._amp_var,
            width=6, format="%.2f"
        )
        amp_spin.grid(row=0, column=4, padx=4)

        # Phase offset (degrees)
        ttk.Label(self, text="Phase (°):").grid(row=0, column=5)
        self._phase_var = tk.DoubleVar(value=0.0)
        phase_spin = ttk.Spinbox(
            self, from_=0, to=360, increment=1, textvariable=self._phase_var, width=6
        )
        phase_spin.grid(row=0, column=6, padx=4)

        ttk.Button(self, text="✕", width=3, command=self._on_remove).grid(
            row=0, column=7, padx=4
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
            self._list_inner, idx, on_remove=lambda i=idx: self._remove_component(i)
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

    # ── Playback ───────────────────────────────────────────────────────────────
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
        self._synthesizer = SynthesizerFrame(notebook)

        notebook.add(self._analyzer,    text="🎤  Analyzer")
        notebook.add(self._synthesizer, text="🎹  Synthesizer")

    def destroy(self) -> None:
        if hasattr(self, "_analyzer"):
            self._analyzer.destroy()
        super().destroy()


def main() -> None:
    app = AcousticsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
