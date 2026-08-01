"""Entry point for the Acoustics application."""

from acoustics.app import AcousticsApp


def main() -> None:
    app = AcousticsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
