"""
Dynamic search visualization.
Generates frame-by-frame animation showing the search process:
- f(K) curve growing as evaluations happen
- Search points appearing with color-coded phases
- Interval shrinking animation for binary/golden/ternary
- Outputs GIF, MP4, or interactive HTML.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from .search import SearchResult


def animate_search(
    result: SearchResult,
    save_path: str = "atcnd_search.gif",
    fps: int = 3,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    show_interval: bool = True,
    show_candidates: bool = True,
    dpi: int = 120,
    format: Optional[str] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if format is None:
        format = save_path.rsplit(".", 1)[-1] if "." in save_path else "gif"

    history = result.search_history
    if not history:
        return

    ks_sorted = sorted(result.all_scores.keys())
    all_ks_set = set()
    all_scores_partial: Dict[int, float] = {}

    phase_colors = {
        "boundary": "#795548",
        "binary_search": "#4CAF50",
        "golden_section": "#FF9800",
        "ternary_search": "#9C27B0",
        "grid": "#607D8B",
        "refinement": "#E91E63",
        "final_sweep": "#00BCD4",
    }

    fig, ax = plt.subplots(figsize=figsize)

    def init():
        ax.clear()
        return []

    def update(frame_idx):
        ax.clear()

        end = min(frame_idx + 1, len(history))
        for i in range(end):
            entry = history[i]
            all_ks_set.add(entry["k"])
            all_scores_partial[entry["k"]] = entry["score"]

        partial_ks = sorted(all_scores_partial.keys())
        partial_scores = [all_scores_partial[k] for k in partial_ks]

        ax.plot(partial_ks, partial_scores, "o-", color="#2196F3",
                linewidth=1.5, markersize=4, alpha=0.5, zorder=2, label="f(K)")

        current_best_k = partial_ks[int(np.argmax(partial_scores))]
        current_best_score = max(partial_scores)
        ax.axvline(x=current_best_k, color="#F44336", linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Best K={current_best_k}")

        if show_interval and end > 0 and result.strategy in ("binary", "golden_section", "ternary"):
            left = history[0]["k"]
            right = history[-1]["k"] if end >= len(history) else history[end - 1]["k"]
            for i in range(end - 1, -1, -1):
                if history[i]["k"] <= current_best_k:
                    left_candidate = history[i]["k"]
                    break
            for i in range(end - 1, -1, -1):
                if history[i]["k"] >= current_best_k:
                    right_candidate = history[i]["k"]
                    break
            ax.axvspan(left_candidate, right_candidate, alpha=0.1, color="#FF9800")

        for i in range(end):
            entry = history[i]
            phase = entry.get("phase", "unknown")
            color = phase_colors.get(phase, "#9E9E9E")
            is_current = (i == end - 1)
            size = 12 if is_current else 7
            zorder = 10 if is_current else 5
            marker = "*" if is_current else "s"
            ax.plot(entry["k"], entry["score"], marker, color=color,
                    markersize=size, zorder=zorder)

        if show_candidates and result.candidate_ks:
            for ck, cs in zip(result.candidate_ks, result.candidate_scores):
                if ck in all_scores_partial:
                    ax.plot(ck, all_scores_partial[ck], "D", color="#E91E63",
                            markersize=8, zorder=8, markeredgecolor="white",
                            markeredgewidth=1.5)

        ax.set_xlabel("K (integer parameter)", fontsize=12)
        ax.set_ylabel("f(K)", fontsize=12)
        strat_name = result.strategy or "search"
        t = title or f"ATCND {strat_name} search - step {end}/{len(history)}"
        ax.set_title(t, fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        x_min = min(result.all_scores.keys()) - 1
        x_max = max(result.all_scores.keys()) + 1
        ax.set_xlim(x_min, x_max)

        if partial_scores:
            y_lo = min(partial_scores) - 0.05 * (max(partial_scores) - min(partial_scores) + 0.01)
            y_hi = max(partial_scores) + 0.05 * (max(partial_scores) - min(partial_scores) + 0.01)
            ax.set_ylim(y_lo, y_hi)

        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(history), interval=1000 // fps,
                         blit=False, repeat=False)

    plt.tight_layout()

    if format in ("gif",):
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    elif format in ("mp4",):
        try:
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)
        except Exception:
            save_path_gif = save_path.rsplit(".", 1)[0] + ".gif"
            anim.save(save_path_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    elif format in ("html", "htm"):
        try:
            anim.save(save_path, writer="html", fps=fps)
        except Exception:
            anim.save(save_path.rsplit(".", 1)[0] + ".gif",
                      writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)

    plt.close(fig)
    return save_path


def animate_search_frames(
    result: SearchResult,
    save_dir: str = ".",
    prefix: str = "frame",
    figsize: tuple = (10, 6),
    dpi: int = 120,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs(save_dir, exist_ok=True)
    paths = []
    history = result.search_history
    phase_colors = {
        "boundary": "#795548", "binary_search": "#4CAF50",
        "golden_section": "#FF9800", "ternary_search": "#9C27B0",
        "grid": "#607D8B", "refinement": "#E91E63", "final_sweep": "#00BCD4",
    }

    partial_scores: Dict[int, float] = {}

    for frame_idx in range(len(history)):
        fig, ax = plt.subplots(figsize=figsize)

        for i in range(frame_idx + 1):
            entry = history[i]
            partial_scores[entry["k"]] = entry["score"]

        sorted_ks = sorted(partial_scores.keys())
        scores = [partial_scores[k] for k in sorted_ks]

        ax.plot(sorted_ks, scores, "o-", color="#2196F3", linewidth=1.5, markersize=4, alpha=0.5)
        current_best_k = sorted_ks[int(np.argmax(scores))]
        ax.axvline(x=current_best_k, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.7)

        for i in range(frame_idx + 1):
            entry = history[i]
            phase = entry.get("phase", "unknown")
            color = phase_colors.get(phase, "#9E9E9E")
            is_current = (i == frame_idx)
            size = 12 if is_current else 7
            marker = "*" if is_current else "s"
            ax.plot(entry["k"], entry["score"], marker, color=color, markersize=size, zorder=10 if is_current else 5)

        ax.set_xlabel("K", fontsize=12)
        ax.set_ylabel("f(K)", fontsize=12)
        strat = result.strategy or "search"
        ax.set_title(f"ATCND {strat} - step {frame_idx + 1}/{len(history)}", fontsize=14)
        ax.grid(True, alpha=0.3)

        path = os.path.join(save_dir, f"{prefix}_{frame_idx:04d}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths