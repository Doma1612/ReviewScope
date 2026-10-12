from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

TODAY = date(2026, 9, 23)

TRACK_COLORS = {
    "Backend & ML": "#2a78d6",
    "Frontend": "#eb6834",
    "All": "#1baf7a",
}
PLAN_COLOR = "#d9d8d4"
TEXT, TEXT_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe8"


def d(s: str) -> date:
    return date.fromisoformat(f"2026-{s}")


# (id, title, track, planned start, planned weeks, actual start, actual end, status)
ROWS = [
    ("WP1", "Setup & infrastructure", "Backend & ML", "05-05", 2, "04-29", "06-28", "Partial: no CI"),
    ("WP2", "Core backend", "Backend & ML", "05-12", 3, "06-01", "07-02", "Done"),
    ("WP3", "Frontend scaffolding & upload", "Frontend", "05-05", 3, "06-01", "06-30", "Done"),
    ("WP4", "Data ingestion (Yelp)", "Backend & ML", "06-01", 2, "05-25", "06-11", "Done"),
    ("WP5", "NLP pipeline: eval + selection", "Backend & ML", "05-19", 6, "05-19", "08-20", "Done"),
    ("WP6", "Cluster visualization", "Frontend", "05-25", 5, "06-15", "08-08", "Done"),
    ("WP7", "RAG engine", "Backend & ML", "07-06", 3, "08-08", "09-22", "Done"),
    ("WP8", "RAG chat interface", "Frontend", "07-06", 3, "08-08", "09-22", "Done, no chat history"),
    ("WP9", "EuroParl integration", "Backend & ML", "07-20", 3, "09-18", "09-22", "Done, English only"),
    ("WP9b", "Reproducibility / incremental", "Backend & ML", "07-20", 4, "06-11", "08-08", "Partial: per-seed only"),
    ("WP10", "Timeline visualization", "Frontend", "07-27", 4, "09-18", "09-21", "Partial: not merged"),
    ("WP11", "Integration & testing", "All", "08-17", 3, "08-20", "09-23", "Ongoing"),
    ("WP12", "Demo prep", "All", "09-07", 2, "09-20", "09-24", "Ongoing"),
]

MILESTONES = [
    ("Plan", "05-07"), ("MVP", "06-29"), ("Midterm", "07-02"),
    ("Final", "09-24"), ("Resources due", "10-12"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    plt.rcParams["font.family"] = "DejaVu Sans"

    n = len(ROWS)
    for i, (wp, title, track, ps, pw, as_, ae, status) in enumerate(ROWS):
        y = n - i
        p0 = d(ps)
        ax.barh(y + 0.17, timedelta(weeks=pw), left=p0, height=0.22,
                color=PLAN_COLOR, edgecolor="none")
        a0, a1 = d(as_), d(ae)
        ax.barh(y - 0.12, a1 - a0 + timedelta(days=1), left=a0, height=0.34,
                color=TRACK_COLORS[track], edgecolor=SURFACE, linewidth=2)
        ax.text(date(2026, 10, 17), y - 0.02, status, va="center", ha="left",
                fontsize=11, color=TEXT_2)
        ax.text(date(2026, 4, 25), y, f"{wp}", va="center", ha="right",
                fontsize=12, color=TEXT, fontweight="bold",
                transform=ax.transData)
        ax.text(date(2026, 4, 25), y - 0.33, title, va="center", ha="right",
                fontsize=9.5, color=TEXT_2)

    top = n + 1
    nudge = {"MVP": ("right", -2), "Midterm": ("left", 2)}
    for name, ds in MILESTONES:
        m = d(ds)
        ha, shift = nudge.get(name, ("center", 0))
        ax.axvline(m, color=GRID, linewidth=1, zorder=0)
        ax.plot(m, top, marker="D", markersize=9, color=TEXT, zorder=3)
        ax.text(m + timedelta(days=shift), top + 0.45, f"{name}\n{m:%d.%m}", ha=ha, va="bottom",
                fontsize=9.5, color=TEXT)

    ax.axvline(TODAY, color=TEXT, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.text(TODAY - timedelta(days=2), top - 0.45, "today", ha="right", va="center",
            fontsize=9.5, color=TEXT)

    ax.set_ylim(0.3, top + 1.6)
    ax.set_xlim(date(2026, 4, 27), date(2026, 10, 15))
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.tick_params(axis="x", colors=TEXT_2, labelsize=11, length=0)
    ax.grid(axis="x", which="major", color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PLAN_COLOR)]
    labels = ["Planned (project plan, May)"]
    for track, color in TRACK_COLORS.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
        labels.append(f"Actual: {track}")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=11, labelcolor=TEXT, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("ReviewScope: plan vs. actual", x=0.03, ha="left",
                 fontsize=18, color=TEXT, fontweight="bold")
    fig.text(0.03, 0.915, "Actual windows reconstructed from commit and run dates",
             fontsize=11, color=TEXT_2)
    fig.subplots_adjust(left=0.19, right=0.85, top=0.86, bottom=0.09)

    out = Path(__file__).resolve().parents[1] / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gantt-final.png", facecolor=SURFACE)
    fig.savefig(out / "gantt-final.svg", facecolor=SURFACE)
    print(out / "gantt-final.png")


if __name__ == "__main__":
    main()
