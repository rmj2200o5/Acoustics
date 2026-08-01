"""Synthesizer tab UI and waveform playback logic."""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .constants import SAMPLE_RATE


class SynthComponent(ttk.Frame):
    """A single row in the synthesizer representing one sine-wave component."""

    def __init__(self, parent: tk.Widget, index: int, on_remove, on_more=None):
        super().__init__(parent)
        self._index = index
        self._on_remove = on_remove
        self._on_more = on_more
        self._build_ui()

    def _build_ui(self) -> None:
        ttk.Label(self, text=f"Wave {self._index}").grid(row=0, column=0, padx=4, pady=2)

        ttk.Label(self, text="Freq (Hz):").grid(row=0, column=1, pady=2)
        self._freq_var = tk.DoubleVar(value=440.0)
        freq_spin = tk.Spinbox(
            self,
            from_=20,
            to=20000,
            increment=1,
            textvariable=self._freq_var,
            width=7,
            background="#313244",
            foreground="#cdd6f4",
            insertbackground="#a6e3a1",
            insertwidth=2,
            relief="sunken",
            borderwidth=1,
        )
        freq_spin.grid(row=0, column=2, padx=4, pady=2)
        freq_spin.bind("<FocusIn>", lambda e: freq_spin.select_range(0, tk.END))

        ttk.Label(self, text="Amplitude:").grid(row=0, column=3, pady=2)
        self._amp_var = tk.DoubleVar(value=1.0)
        amp_spin = ttk.Spinbox(
            self, from_=0.0, to=1.0, increment=0.05, textvariable=self._amp_var, width=6, format="%.2f"
        )
        amp_spin.grid(row=0, column=4, padx=4, pady=2)

        ttk.Label(self, text="Phase (°):").grid(row=0, column=5, pady=2)
        self._phase_var = tk.DoubleVar(value=0.0)
        phase_spin = ttk.Spinbox(
            self, from_=0, to=360, increment=1, textvariable=self._phase_var, width=6
        )
        phase_spin.grid(row=0, column=6, padx=4, pady=2)

        more_btn = ttk.Button(self, text="⋯", width=2, command=self._show_more_menu)
        more_btn.grid(row=0, column=7, padx=1, pady=2)

        ttk.Button(self, text="▼", width=2, command=self._octave_down).grid(row=0, column=8, padx=1, pady=2)
        ttk.Button(self, text="▲", width=2, command=self._octave_up).grid(row=0, column=9, padx=1, pady=2)

        ttk.Button(self, text="✕", width=3, command=self._on_remove).grid(row=0, column=10, padx=4, pady=2)

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
        try:
            current_freq = float(self._freq_var.get())
            new_freq = min(20000, current_freq * 2.0)
            self._freq_var.set(new_freq)
        except tk.TclError:
            pass

    def _octave_down(self) -> None:
        try:
            current_freq = float(self._freq_var.get())
            new_freq = max(20, current_freq / 2.0)
            self._freq_var.set(new_freq)
        except tk.TclError:
            pass

    def _show_more_menu(self) -> None:
        if self._on_more is None:
            return

        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Add Harmonics", command=lambda: self._on_more("harmonics", self._index))
        menu.post(self.winfo_pointerx(), self.winfo_pointery())


class SynthesizerFrame(ttk.Frame):
    """Tab 2 – Build a waveform from sine-wave components and play it."""

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._components: list[SynthComponent] = []
        self._component_counter = 0
        self._playback_active = False
        self._build_ui()

    def _build_ui(self) -> None:
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(ctrl, text="➕  Add Sine Wave", command=self._add_component).pack(side=tk.LEFT, padx=4)
        self._btn_play = ttk.Button(ctrl, text="▶  Play", command=self._toggle_playback)
        self._btn_play.pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl, text="🗑  Clear All", command=self._clear_all_components).pack(side=tk.LEFT, padx=4)

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

        ttk.Label(ctrl, text="Duration (s):").pack(side=tk.LEFT, padx=(12, 2))
        self._duration_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(
            ctrl, from_=0.5, to=30.0, increment=0.5, textvariable=self._duration_var, width=5, format="%.1f"
        ).pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(ctrl, textvariable=self._status_var).pack(side=tk.LEFT, padx=12)

        list_outer = ttk.LabelFrame(self, text="Sine-Wave Components")
        list_outer.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

        canvas_scroll = tk.Canvas(list_outer, height=220)
        scrollbar = ttk.Scrollbar(list_outer, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self._list_inner = ttk.Frame(canvas_scroll)
        self._list_inner.bind(
            "<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        canvas_scroll.create_window((0, 0), window=self._list_inner, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._canvas_scroll = canvas_scroll
        canvas_scroll.bind("<MouseWheel>", self._on_mousewheel)
        canvas_scroll.bind("<Button-4>", self._on_mousewheel)
        canvas_scroll.bind("<Button-5>", self._on_mousewheel)

        notes_frame = ttk.LabelFrame(self, text="Quick Add: Standard Notes (A=440Hz)")
        notes_frame.pack(fill=tk.X, padx=8, pady=4)
        self._note_freqs = {
            "C": 261.63,
            "C#": 277.18,
            "D": 293.66,
            "D#": 311.13,
            "E": 329.63,
            "F": 349.23,
            "F#": 369.99,
            "G": 391.99,
            "G#": 415.30,
            "A": 440.00,
            "A#": 466.16,
            "B": 493.88,
        }
        notes_inner = ttk.Frame(notes_frame)
        notes_inner.pack(fill=tk.X, padx=4, pady=4)
        for note_name, freq in self._note_freqs.items():
            ttk.Button(
                notes_inner, text=note_name, width=4, command=lambda f=freq: self._add_frequency(f)
            ).pack(side=tk.LEFT, padx=1)

        plot_frame = ttk.LabelFrame(self, text="Waveform Preview")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        fig = Figure(figsize=(9, 2.5), dpi=96, facecolor="#1e1e2e")
        self._ax = fig.add_subplot(111)
        self._setup_axes()
        (self._line,) = self._ax.plot([], [], color="#a6e3a1", lw=1.2)

        self._canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

    def _add_component(self) -> None:
        self._component_counter += 1
        idx = self._component_counter
        comp = SynthComponent(
            self._list_inner,
            idx,
            on_remove=lambda i=idx: self._remove_component(i),
            on_more=self._handle_component_option,
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
        if amplitudes is None:
            amplitudes = np.ones_like(frequencies)

        for freq, amp in zip(frequencies, amplitudes):
            if freq > 0:
                self._component_counter += 1
                idx = self._component_counter
                comp = SynthComponent(
                    self._list_inner,
                    idx,
                    on_remove=lambda i=idx: self._remove_component(i),
                    on_more=self._handle_component_option,
                )
                comp.pack(fill=tk.X, pady=2, padx=4)
                self._components.append(comp)
                comp._freq_var.set(float(freq))
                comp._amp_var.set(float(amp))
        self._refresh_preview()

    def _clear_all_components(self) -> None:
        for comp in list(self._components):
            self._components.remove(comp)
            comp.destroy()
        self._canvas_scroll.configure(scrollregion=self._canvas_scroll.bbox("all"))
        self._refresh_preview()

    def _add_frequency(self, frequency: float) -> None:
        self.add_frequencies(np.array([frequency]), np.array([1.0]))

    def _handle_component_option(self, option: str, comp_index: int) -> None:
        if option == "harmonics":
            self._add_harmonics_dialog(comp_index)

    def _add_harmonics_dialog(self, comp_index: int) -> None:
        comp = next((c for c in self._components if c._index == comp_index), None)
        if comp is None:
            return

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
                    harmonic_amp = fundamental_amp / (harmonic_num**2)
                    harmonics_freqs.append(harmonic_freq)
                    harmonics_amps.append(harmonic_amp)

                if harmonics_freqs:
                    self.add_frequencies(np.array(harmonics_freqs), np.array(harmonics_amps))

                dialog.destroy()
                messagebox.showinfo("Success", f"Added {len(harmonics_freqs)} harmonics")
            except (ValueError, tk.TclError):
                messagebox.showerror("Error", "Invalid input")

        ttk.Button(dialog, text="Add", command=add_harmonics).pack(pady=10)

    def _sort_components(self, sort_by: str) -> None:
        if sort_by == "frequency":
            self._components.sort(key=lambda c: c.frequency)
        elif sort_by == "amplitude":
            self._components.sort(key=lambda c: c.amplitude, reverse=True)
        elif sort_by == "order":
            self._components.sort(key=lambda c: c._index)

        for comp in self._components:
            comp.pack_forget()

        for comp in self._components:
            comp.pack(fill=tk.X, pady=2, padx=4)

        self._list_inner.update_idletasks()
        self._canvas_scroll.update_idletasks()
        self._canvas_scroll.configure(scrollregion=self._canvas_scroll.bbox("all"))
        self._refresh_preview()

    def _build_waveform(self, duration: float) -> np.ndarray:
        n_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        signal = np.zeros(n_samples, dtype=np.float64)
        for comp in self._components:
            signal += comp.amplitude * np.sin(2.0 * np.pi * comp.frequency * t + comp.phase_rad)
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal /= peak
        return signal.astype(np.float32)

    def _refresh_preview(self) -> None:
        preview_duration = 0.05
        signal = self._build_waveform(preview_duration)
        t_ms = np.linspace(0, preview_duration * 1000, len(signal), endpoint=False)
        self._line.set_xdata(t_ms)
        self._line.set_ydata(signal)
        self._ax.set_xlim(0, preview_duration * 1000)
        self._ax.set_ylim(-1.1, 1.1)
        self._canvas.draw_idle()

    def _on_mousewheel(self, event) -> None:
        if event.num == 5 or event.delta < 0:
            self._canvas_scroll.yview_scroll(3, "units")
        else:
            self._canvas_scroll.yview_scroll(-3, "units")

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
