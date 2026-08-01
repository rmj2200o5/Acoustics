"""Main Tk application window and notebook wiring."""

import tkinter as tk
from tkinter import ttk

from .analyzer import AnalyzerFrame
from .synthesizer import SynthesizerFrame
from .tuner import TunerFrame


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

        self._analyzer = AnalyzerFrame(notebook)
        self._tuner = TunerFrame(notebook)
        self._synthesizer = SynthesizerFrame(notebook)

        self._analyzer._synthesizer_frame = self._synthesizer

        notebook.add(self._analyzer, text="🎤  Analyzer")
        notebook.add(self._tuner, text="🎵  Tuner")
        notebook.add(self._synthesizer, text="🎹  Synthesizer")

        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event) -> None:
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
