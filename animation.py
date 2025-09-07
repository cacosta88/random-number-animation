"""
Animated normal sampling demo
- Streams random draws from N(mu, sigma) and updates a histogram
- Overlays the theoretical Normal PDF scaled to histogram counts
- No external dependencies beyond numpy + matplotlib

Run: python this_file.py
In notebooks: just run the cell; the animation will play inline (with %matplotlib notebook) or in a pop-up.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_normal(mu=0.0, sigma=1.0, n_samples=5000, bins=40, batch=50, interval_ms=60, xlim=None):
    """
    Parameters
    ----------
    mu, sigma : float
        Target normal distribution parameters.
    n_samples : int
        Total number of samples to draw.
    bins : int
        Number of histogram bins.
    batch : int
        Samples added per animation frame.
    interval_ms : int
        Delay between frames (milliseconds).
    xlim : (float, float) or None
        X-axis limits. Default: (mu - 4*sigma, mu + 4*sigma).

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The constructed animation object.
    """
    rng = np.random.default_rng()
    if xlim is None:
        xlim = (mu - 4.0 * sigma, mu + 4.0 * sigma)

    # Pre-generate all draws (faster & deterministic for a given seed)
    data = rng.normal(mu, sigma, size=n_samples)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fixed bin edges so we can update bar heights in-place
    binedges = np.linspace(xlim[0], xlim[1], bins + 1)
    binwidth = binedges[1] - binedges[0]
    bincenters = 0.5 * (binedges[:-1] + binedges[1:])

    # Initialize histogram bars at height 0
    bars = ax.bar(bincenters, np.zeros_like(bincenters), width=binwidth, align="center", edgecolor="black")

    # Theoretical normal curve shape (unit area PDF)
    x = np.linspace(xlim[0], xlim[1], 400)
    pdf_shape = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))

    # Line for the scaled PDF (scaled to expected counts)
    (line,) = ax.plot(x, np.zeros_like(x), linewidth=2)

    # Live stats text
    stats_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", ha="left")

    # Axes cosmetics
    ax.set_xlim(*xlim)
    # Max expected bin height happens near the mode; add padding
    max_expected = n_samples * binwidth * pdf_shape.max()
    ax.set_ylim(0, max_expected * 1.2)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title("Accumulating random draws → histogram approaches Normal(μ, σ)")

    def init():
        for b in bars:
            b.set_height(0.0)
        line.set_data(x, np.zeros_like(x))
        stats_text.set_text("")
        return (*bars, line, stats_text)

    def update(frame_idx):
        k = min((frame_idx + 1) * batch, n_samples)

        # Update histogram heights for the first k samples
        counts, _ = np.histogram(data[:k], bins=binedges)
        for b, h in zip(bars, counts):
            b.set_height(h)

        # Scale the theoretical PDF to match counts: expected_count = k * binwidth * pdf
        line.set_data(x, pdf_shape * k * binwidth)

        # Live sample stats
        sample = data[:k]
        stats_text.set_text(f"n = {k}\nmean = {sample.mean():.3f}\nstd = {sample.std(ddof=1):.3f}")

        return (*bars, line, stats_text)

    frames = int(np.ceil(n_samples / batch))
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval_ms,
        blit=True,
    )
    return ani

if __name__ == "__main__":
    # Tweak these if you like
    ani = animate_normal(mu=0, sigma=1, n_samples=6000, bins=45, batch=60, interval_ms=50)

    # To display in a Python script:
    plt.show()

    # --- Optional: save to GIF or MP4 (uncomment one block below) ---
    # Requires pillow:
    # ani.save("normal_animation.gif", writer="pillow", fps=int(1000/50))
    # Requires ffmpeg:
    # ani.save("normal_animation.mp4", writer="ffmpeg", fps=int(1000/50))
