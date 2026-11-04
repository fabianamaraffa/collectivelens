"""Entry point for the web app (also used by Hugging Face Spaces).

Run locally with:  python app.py   or   python -m collectivelens app
"""

from collectivelens.ui import build_app, launch_kwargs

demo = build_app()

if __name__ == "__main__":
    demo.launch(**launch_kwargs())
