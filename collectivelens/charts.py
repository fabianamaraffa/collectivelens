"""Charts for the Insights tab.

Colours follow a fixed order, so a topic keeps its colour whatever is filtered. The palette is the
validated eight-hue categorical set from the data-viz guidance (adjacent pairs stay distinguishable
for the common kinds of colour blindness). Where colour alone could be ambiguous (the library map)
each topic also has its own marker shape. Figures are drawn on a light card so they read in both
the light and dark app themes.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .embedders import as_dense
from .library import Library

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#eaf1fb", "#9cc0ee", "#2a78d6", "#123f7a"])


def _figure(width: float = 7.4, height: float = 4.2) -> tuple[Figure, object]:
    fig = Figure(figsize=(width, height), dpi=130, facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    return fig, ax


def _empty(message: str) -> Figure:
    fig, ax = _figure(7.4, 2.2)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=INK_2, fontsize=10, transform=ax.transAxes)
    return fig


def _wrap(label: str, width: int = 30) -> str:
    return label if len(label) <= width else label[: width - 1].rstrip() + "…"


def topic_color(lib: Library, name: str) -> str:
    return CATEGORICAL[lib.topic_names.index(name) % len(CATEGORICAL)]


def topic_bar(lib: Library) -> Figure:
    """Documents per topic, one hue (the bars are labelled, so colour carries no extra meaning)."""
    counts = {n: 0 for n in lib.topic_names}
    for t in lib.topics:
        counts[t.name] += 1
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = _figure(7.4, 0.42 * len(items) + 1.2)
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], color=CATEGORICAL[0], height=0.62)
    ax.set_yticks(y, [_wrap(k) for k, _ in items], color=INK)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    top = max(v for _, v in items) or 1
    for yi, (_, v) in zip(y, items):
        ax.text(v + top * 0.015, yi, str(v), va="center", color=INK, fontsize=9)
    ax.set_xlim(0, top * 1.12)
    ax.set_title("Documents per topic", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def year_topic_heatmap(lib: Library) -> Figure:
    """How many documents each topic has per year (darker is more)."""
    years = lib.years()
    if len(years) < 2:
        return _empty("The heat map needs documents from at least two different years.")
    names = lib.topic_names
    grid = np.zeros((len(names), len(years)), dtype=int)
    for doc, topic in zip(lib.docs, lib.topics):
        if doc.year:
            grid[names.index(topic.name), years.index(doc.year)] += 1
    fig, ax = _figure(7.4, 0.42 * len(names) + 1.5)
    ax.imshow(grid, cmap=SEQUENTIAL, aspect="auto", vmin=0, vmax=max(grid.max(), 1))
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_yticks(range(len(names)), [_wrap(n) for n in names], color=INK)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(years)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    peak = max(grid.max(), 1)
    for r in range(len(names)):
        for c in range(len(years)):
            if grid[r, c]:
                ax.text(c, r, grid[r, c], ha="center", va="center", fontsize=9,
                        color="#ffffff" if grid[r, c] > peak * 0.55 else INK)
    ax.set_title("Topics by publication year", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def org_bar(lib: Library, limit: int = 12) -> Figure:
    counts: dict[str, int] = {}
    for d in lib.docs:
        counts[d.org] = counts.get(d.org, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit][::-1]
    fig, ax = _figure(7.4, 0.42 * len(items) + 1.2)
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], color=CATEGORICAL[0], height=0.62)
    ax.set_yticks(y, [_wrap(k) for k, _ in items], color=INK)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    top = max(v for _, v in items) or 1
    for yi, (_, v) in zip(y, items):
        ax.text(v + top * 0.015, yi, str(v), va="center", color=INK, fontsize=9)
    ax.set_xlim(0, top * 1.12)
    ax.set_title("Documents per organisation", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def library_map(lib: Library) -> Figure:
    """Every document as a point; documents about similar things sit close together.

    A projection of many dimensions onto two loses information, so distances are only a rough guide.
    """
    if len(lib.docs) < 3:
        return _empty("The map needs at least three documents.")
    xy = PCA(n_components=2, random_state=0).fit_transform(as_dense(lib.doc_vecs))
    fig, ax = _figure(7.4, 5.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.margins(0.12)
    for i, name in enumerate(lib.topic_names):
        idx = [j for j, t in enumerate(lib.topics) if t.name == name]
        if not idx:
            continue
        ax.scatter(xy[idx, 0], xy[idx, 1], s=80, color=CATEGORICAL[i % 8], marker=MARKERS[i % 8],
                   edgecolor=SURFACE, linewidth=1.2, label=_wrap(name, 40), zorder=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK, handletextpad=0.4, columnspacing=1.6)
    ax.set_title("Library map (each point is a document)", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig
