import sys
import subprocess

from pathlib import Path

APP_PATH = Path(__file__).parent / "app.py"


def run():
    args = sys.argv[1:]  # forward any additional arguments to streamlit run
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)] + args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run()
