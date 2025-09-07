Animated Normal Sampling


A tiny Python demo that animates random draws from a Normal(μ, σ) and shows how the histogram converges to the bell curve. Built with NumPy + Matplotlib only.


https://github.com/your-org/your-repo
 (replace with your repo)



✨ What it does

Streams random samples in batches and updates a histogram in real time

Overlays the theoretical Normal PDF scaled to counts (so heights are comparable)

Displays live stats (n, sample mean, sample std)



🧰 Requirements

Python 3.8+
numpy
matplotlib
(optional) pillow for saving GIFs, or ffmpeg for MP4

Create a quick requirements.txt:
numpy
matplotlib
pillow   # optional, for GIF export


Install:
pip install -r requirements.txt


🚀 Quickstart

Save the script as animated_normal.py (or use the provided file if this repo includes it), then:

python animated_normal.py



You’ll see a window with the live histogram and the overlaid bell curve.
