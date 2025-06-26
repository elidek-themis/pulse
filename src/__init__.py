import sys
import subprocess

from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def run():
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    run()
