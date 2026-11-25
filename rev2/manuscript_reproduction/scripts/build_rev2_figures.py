"""Build all four new KAIS Revision 2.0 publication figures (Fig5, Fig7, Fig8, Fig10).

Matches the Springer KAIS paper_style.py (174mm width, Arial 8.5pt, vector Type 42 PDF,
Okabe-Ito colorblind-safe palette, strict zero-canvas-text-clipping audit).
"""
import csv
import sys
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as ps

ROOT = HERE.parent
DATA = ROOT / "supplementary_data"
OUT = ROOT / "figures"

DATASETS = ["ml1m", "beauty", "sports", "toys"]
DS_TITLES = {
    "ml1m": "(a) MovieLens-1M",
    "beauty": "(b) Beauty",
    "sports": "(c) Sports",
    "toys": "(d) Toys",
}

# Okabe-Ito colorblind-safe paradigm palette
PARADIGM = {
    "sasrec":    dict(name="SASRec",    color="#C0392B", marker="o", grp="attn"),
    "lsab":      dict(name="LSAB",      color="#C0392B", marker="s", grp="attn"),
    "ticoserec": dict(name="TiCoSeRec", color="#C0392B", marker="^", grp="attn"),
    "tisasrec":  dict(name="TiSASRec",  color="#D55E00", marker="D", grp="time"),
    "fmlprec":   dict(name="FMLP-Rec",  color="#0072B2", marker="s", grp="freq"),
    "bsarec":    dict(name="BSARec",    color="#0072B2", marker="P", grp="freq"),
    "lrurec":    dict(name="LRURec",    color="#00895E", marker="^", grp="ssm"),
    "gru4rec":   dict(name="GRU4Rec",   color="#555555", marker="o", grp="rnn", hollow=True),
}


def load_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def grid(ax, axis="both"):
    ax.grid(axis=axis, color="#e0e0e0", lw=0.45, ls="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, pad=2.5)


def save_audited(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w_px, h_px = fig.get_size_inches() * fig.dpi
    for txt in fig.findobj(match=lambda o: hasattr(o, "get_text")):
        if not getattr(txt, "get_visible", lambda: False)() or not txt.get_text():
            continue
        box = txt.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        if min(box.x0, box.y0) < -1.5 or box.x1 > w_px + 1.5 or box.y1 > h_px + 1.5:
            raise ValueError(f"Canvas clipping in {name}: '{txt.get_text()}' bbox={box.bounds} vs ({w_px:.1f},{h_px:.1f})")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches=None, pad_inches=0)
    ps.save(fig, OUT / f"{name}.pdf")
    print(f"[OK] Saved {name}.pdf & {name}.png (zero clipping)")


def build_fig5(master):
    """Fig5: 8-Backbone Accuracy vs. Last-Bucket Stability across 4 datasets."""
    idx = {(r["dataset"], r["backbone"], r["pooling"]): r for r in master}
    fig = ps.figure(110)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.095, top=0.845, hspace=0.38, wspace=0.22)

    handles = [
        Line2D([0], [0], label="SASRec / LSAB / TiCoSeRec (Self-Attn)", marker="o", color="#C0392B", ls="None", ms=4.5),
        Line2D([0], [0], label="TiSASRec (Time-Aware)", marker="D", color="#D55E00", ls="None", ms=4.3),
        Line2D([0], [0], label="FMLP-Rec / BSARec (Freq.)", marker="s", color="#0072B2", ls="None", ms=4.5),
        Line2D([0], [0], label="LRURec (Linear SSM)", marker="^", color="#00895E", ls="None", ms=4.8),
        Line2D([0], [0], label="GRU4Rec (RNN)", marker="o", color="#555555", mfc="white", mew=1.2, ls="None", ms=4.5),
        Line2D([0], [0], label="100% Exact Invariance (LBSP)", color="#00895E", ls="--", lw=1.0),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.53, 0.992), fontsize=7.4, handletextpad=0.4, columnspacing=1.2
    )

    # Carefully tuned non-overlapping text offsets (dx_pt, dy_pt, ha, va, use_arrow)
    OFFSETS = {
        "ml1m": {
            "tisasrec":  (11,  8, "left",  "bottom", True),
            "sasrec":    (-8,  9, "right", "bottom", True),
            "lsab":      (-8, -7, "right", "top",    True),
            "ticoserec": (13,  4, "left",  "center", True),
            "bsarec":    (12, -4, "left",  "top",    True),
            "fmlprec":   (-9, -9, "right", "top",    True),
            "lrurec":    (10,-13, "left",  "top",    True),
            "gru4rec":   (8,   2, "left",  "center", False),
        },
        "beauty": {
            "fmlprec":   (8,   4, "left",  "bottom", False),
            "bsarec":    (8,  -4, "left",  "top",    False),
            "tisasrec":  (-6, 15, "center","bottom", True),
            "sasrec":    (-7, 11, "right", "bottom", True),
            "lsab":      (-7,-12, "right", "top",    True),
            "ticoserec": (8, -16, "left",  "top",    True),
            "lrurec":    (8,   0, "left",  "center", False),
            "gru4rec":   (8,   0, "left",  "center", False),
        },
        "sports": {
            "fmlprec":   (-8,  7, "right", "bottom", True),
            "bsarec":    (8,   6, "left",  "bottom", True),
            "tisasrec":  (8,   0, "left",  "center", False),
            "sasrec":    (-7, 11, "right", "bottom", True),
            "ticoserec": (9, -12, "left",  "top",    True),
            "lsab":      (-7,-12, "right", "top",    True),
            "lrurec":    (8,   0, "left",  "center", False),
            "gru4rec":   (8,   0, "left",  "center", False),
        },
        "toys": {
            "bsarec":    (8,   4, "left",  "bottom", False),
            "tisasrec":  (-7,  7, "right", "bottom", True),
            "fmlprec":   (8,  -3, "left",  "top",    False),
            "ticoserec": (6, -18, "left",  "top",    True),
            "sasrec":    (-7, 10, "right", "bottom", True),
            "lsab":      (-7,-11, "right", "top",    True),
            "lrurec":    (8,   0, "left",  "center", False),
            "gru4rec":   (8,   0, "left",  "center", False),
        },
    }

    LIMITS = {
        "ml1m":   ((59, 103), (41.0, 53.5), [60, 70, 80, 90, 100]),
        "beauty": ((31, 103), (3.0,  12.6), [40, 55, 70, 85, 100]),
        "sports": ((31, 103), (4.0,  13.4), [40, 55, 70, 85, 100]),
        "toys":   ((31, 103), (4.0,  13.2), [40, 55, 70, 85, 100]),
    }

    for ax, ds in zip(axes.flat, DATASETS):
        xlim, ylim, xticks = LIMITS[ds]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t}%" for t in xticks])
        grid(ax, "both")
        ax.set_title(DS_TITLES[ds], loc="left", fontweight="bold", pad=4)

        ax.axvline(100.0, color="#00895E", ls="--", lw=0.95, zorder=1)

        for bb, st in PARADIGM.items():
            r = idx[(ds, bb, "none")]
            x = float(r["lb_affected_overlap_mean"]) * 100
            xs = float(r["lb_affected_overlap_sd"]) * 100
            y = float(r["complete_ndcg10_mean"]) * 1000
            ys = float(r["complete_ndcg10_sd"]) * 1000

            mfc = "white" if st.get("hollow") else st["color"]
            mew = 1.2 if st.get("hollow") else 0.6
            mec = st["color"] if st.get("hollow") else "white"

            ax.errorbar(
                x, y, xerr=xs, yerr=ys, fmt="none", ecolor=st["color"],
                alpha=0.35, elinewidth=0.75, capsize=1.6, zorder=2
            )
            ax.plot(
                x, y, marker=st["marker"], color=st["color"], mfc=mfc,
                mec=mec, mew=mew, ms=5.0, ls="None", zorder=4
            )

            dx, dy, ha, va, arr = OFFSETS[ds][bb]
            ap = dict(arrowstyle="-", lw=0.55, color="#666666", shrinkA=0, shrinkB=2.5) if arr else None
            ax.annotate(
                st["name"], xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=6.9, color="#222222", arrowprops=ap, zorder=5
            )

    fig.text(0.54, 0.018, "Affected-user Last-Bucket Top-10 Overlap (%)", ha="center", fontsize=8.5)
    fig.text(0.020, 0.47, r"Complete-Panel NDCG@10 ($\times10^{-3}$)", va="center", ha="center", rotation=90, fontsize=8.5)
    save_audited(fig, "Fig5")


def build_fig7(master, contrasts):
    """Fig7: Cross-backbone plug-in LBSP gains across 5 backbones x 4 datasets (2x2 dumbbells)."""
    m_idx = {(r["dataset"], r["backbone"], r["pooling"]): r for r in master}
    c_idx = {(r["dataset"], r["backbone"], r["pooling"]): r for r in contrasts}

    fig = ps.figure(108)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.10, top=0.855, hspace=0.40, wspace=0.34)

    handles = [
        Line2D([0], [0], label="Unpooled Baseline (None)", marker="o", color="#444444", mfc="white", mew=1.3, ls="None", ms=5.0),
        Line2D([0], [0], label="+LBSP Gain (100% LB Inv.)", marker="o", color="#00895E", ls="-", lw=1.8, ms=5.2),
        Line2D([0], [0], label="+LBSP Neutral Dip (100% LB Inv.)", marker="o", color="#D55E00", ls="-", lw=1.8, ms=5.2),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 0.990), fontsize=7.8, columnspacing=1.5)

    ORDER = {
        "ml1m":   ["sasrec", "tisasrec", "gru4rec", "ticoserec", "bsarec"],
        "beauty": ["tisasrec", "bsarec", "ticoserec", "sasrec", "gru4rec"],
        "sports": ["bsarec", "tisasrec", "sasrec", "ticoserec", "gru4rec"],
        "toys":   ["tisasrec", "bsarec", "sasrec", "ticoserec", "gru4rec"],
    }
    BBL = {"sasrec": "SASRec", "tisasrec": "TiSASRec", "gru4rec": "GRU4Rec", "bsarec": "BSARec", "ticoserec": "TiCoSeRec"}
    XLIM = {
        "ml1m":   ((43.5, 62.5), [44, 48, 52, 56]),
        "beauty": ((4.0,  14.2), [4, 6, 8, 10]),
        "sports": ((5.0,  15.6), [6, 8, 10, 12]),
        "toys":   ((4.5,  15.8), [5, 7, 9, 11, 13]),
    }

    for ax, ds in zip(axes.flat, DATASETS):
        bbs = ORDER[ds]
        ys = np.arange(len(bbs))[::-1]
        xlim, xticks = XLIM[ds]
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(-0.55, len(bbs) - 0.35)
        ax.set_yticks(ys)
        ax.set_yticklabels([BBL[b] for b in bbs], fontsize=7.8)
        grid(ax, "x")
        ax.set_title(DS_TITLES[ds], loc="left", fontweight="bold", pad=4)

        x_annot = xlim[1] - 0.015 * (xlim[1] - xlim[0])

        for y, bb in zip(ys, bbs):
            mb = m_idx[(ds, bb, "none")]
            ml = m_idx[(ds, bb, "lbsp_mean")]
            cc = c_idx[(ds, bb, "lbsp_mean")]
            vb = float(mb["complete_ndcg10_mean"]) * 1000
            vl = float(ml["complete_ndcg10_mean"]) * 1000
            d = float(cc["delta_complete_ndcg10_e3"])
            lo = float(cc["delta_complete_ci_lo_e3"])
            ab_b = float(mb["ab_affected_overlap_mean"]) * 100
            ab_l = float(ml["ab_affected_overlap_mean"]) * 100

            col = "#00895E" if vl >= vb else "#D55E00"
            ax.plot([vb, vl], [y, y], color=col, lw=1.9, zorder=2)
            ax.plot(vb, y, marker="o", color="#444444", mfc="white", mew=1.2, ms=4.8, zorder=3)
            ax.plot(vl, y, marker="o", color=col, mec="white", mew=0.6, ms=5.3, zorder=4)

            star = "*" if lo > 0 else ""
            lbl = f"{d:+.2f}{star}"
            ax.text(x_annot, y + 0.14, lbl, ha="right", va="center", fontsize=7.2, fontweight="bold", color=col)
            ax.text(x_annot, y - 0.18, f"AB {ab_b:.0f}→{ab_l:.0f}%", ha="right", va="center", fontsize=6.1, color="#555555")

    fig.text(0.54, 0.018, r"Complete-Panel NDCG@10 ($\times10^{-3}$; right labels: $\Delta$NDCG@10 and All-Bucket Overlap shift)", ha="center", fontsize=8.2)
    save_audited(fig, "Fig7")


def build_fig8(master, contrasts):
    """Fig8: Pooling operator ablation (top row) & LB vs AB overlap (bottom row) on SASRec."""
    m_idx = {(r["dataset"], r["backbone"], r["pooling"]): r for r in master}
    c_idx = {(r["dataset"], r["backbone"], r["pooling"]): r for r in contrasts}

    MODES = [
        ("none", "None", "#7F8C8D"),
        ("lbsp_mean", "Mean", "#0072B2"),
        ("lbsp_max", "Max", "#7B61A8"),
        ("lbsp_attn", "Attn", "#9B59B6"),
        ("lbsp_deepsets", "Deep", "#CC79A7"),
        ("absp_mean", "ABSP", "#00895E"),
    ]

    fig = ps.figure(100)
    axes = fig.subplots(2, 4)
    fig.subplots_adjust(left=0.078, right=0.990, bottom=0.125, top=0.845, hspace=0.40, wspace=0.26)

    top_handles = [
        Patch(facecolor="#7F8C8D", label="None (Control)"),
        Patch(facecolor="#0072B2", label="LBSP-Mean"),
        Patch(facecolor="#7B61A8", label="LBSP-Max / Attn / DeepSets"),
        Patch(facecolor="#00895E", label="ABSP-Mean (Global)"),
        Patch(facecolor="#56B4E9", hatch="///", edgecolor="#222222", lw=0.4, label="AB Overlap (bottom row)"),
    ]
    fig.legend(handles=top_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.992), fontsize=7.3, columnspacing=1.2)

    YLIM_TOP = {
        "ml1m":   (44, 58.0),
        "beauty": (5.0, 9.4),
        "sports": (6.8, 10.6),
        "toys":   (6.0, 11.4),
    }

    x = np.arange(len(MODES))
    for col_i, ds in enumerate(DATASETS):
        # Top row: Full NDCG@10 bars + SD error bars
        ax_t = axes[0, col_i]
        ax_t.set_title(DS_TITLES[ds], loc="left", fontweight="bold", pad=4, fontsize=8.3)
        vals = [float(m_idx[(ds, "sasrec", m)]["complete_ndcg10_mean"]) * 1000 for m, _, _ in MODES]
        sds = [float(m_idx[(ds, "sasrec", m)]["complete_ndcg10_sd"]) * 1000 for m, _, _ in MODES]
        cols = [c for _, _, c in MODES]

        ax_t.axhline(vals[0], color="#666666", ls="--", lw=0.75, zorder=1)
        ax_t.bar(x, vals, width=0.64, color=cols, alpha=0.90, zorder=2)
        ax_t.errorbar(x, vals, yerr=sds, fmt="none", ecolor="#222222", elinewidth=0.65, capsize=1.6, zorder=3)

        # Mark 3-seed CI > 0 with *
        for xi, (m, _, _) in enumerate(MODES):
            if m != "none":
                lo = float(c_idx[(ds, "sasrec", m)]["delta_complete_ci_lo_e3"])
                if lo > 0:
                    ax_t.text(xi, vals[xi] + sds[xi] + 0.03 * (YLIM_TOP[ds][1] - YLIM_TOP[ds][0]), "*",
                              ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#00895E")

        ax_t.set_ylim(*YLIM_TOP[ds])
        ax_t.set_xticks(x)
        ax_t.set_xticklabels([])
        grid(ax_t, "y")

        # Bottom row: LB Overlap (solid) vs AB Overlap (hatched)
        ax_b = axes[1, col_i]
        lb_ov = [float(m_idx[(ds, "sasrec", m)]["lb_affected_overlap_mean"]) * 100 for m, _, _ in MODES]
        ab_ov = [float(m_idx[(ds, "sasrec", m)]["ab_affected_overlap_mean"]) * 100 for m, _, _ in MODES]
        w = 0.36
        ax_b.bar(x - w / 2, lb_ov, width=w, color=cols, alpha=0.92, zorder=2)
        ax_b.bar(x + w / 2, ab_ov, width=w, color="#56B4E9", hatch="///", edgecolor="#333333", lw=0.35, alpha=0.88, zorder=2)
        ax_b.axhline(100.0, color="#00895E", ls=":", lw=0.75, zorder=1)
        ax_b.set_ylim(35, 108)
        ax_b.set_yticks([40, 60, 80, 100])
        if col_i > 0:
            ax_b.set_yticklabels([])
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([lbl for _, lbl, _ in MODES], fontsize=7.0, rotation=32, ha="right")
        grid(ax_b, "y")

    axes[0, 0].set_ylabel(r"Full NDCG@10 ($\times10^{-3}$)", fontsize=8.0)
    axes[1, 0].set_ylabel("Top-10 Overlap (%)", fontsize=8.0)
    fig.text(0.54, 0.015, "SASRec set-pooling variant (solid bar = Last-Bucket Overlap; hatched bar = All-Bucket Overlap)", ha="center", fontsize=8.0)
    save_audited(fig, "Fig8")


def build_fig10(costs):
    """Fig10: Unified RTX 4090 cost profile (Inference latency/VRAM & Training epoch time)."""
    c_idx = {(r["dataset"], r["backbone"], r["pooling_mode"]): r for r in costs}
    fig = ps.figure(76)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.145, right=0.975, bottom=0.16, top=0.85, wspace=0.40)

    # Panel (a): Sports inference latency vs peak VRAM (log y-axis for clean separation)
    ax1 = axes[0]
    ax1.set_title("(a) Sports Inference Latency vs. Peak VRAM", loc="left", fontweight="bold", pad=5, fontsize=8.3)
    ITEMS_A = [
        ("gru4rec",  "none",          "GRU4Rec",      "#555555", "o", (0,   16, "center")),
        ("fmlprec",  "none",          "FMLP-Rec",     "#0072B2", "s", (10,  22, "left")),
        ("sasrec",   "none",          "SASRec",       "#C0392B", "o", (-6, -15, "right")),
        ("sasrec",   "absp_mean",     "ABSP-Mean",    "#00895E", "^", (3,  -19, "center")),
        ("sasrec",   "lbsp_mean",     "LBSP-Mean",    "#00895E", "D", (12,  -4, "left")),
        ("sasrec",   "lbsp_max",      "LBSP-Max",     "#7B61A8", "s", (8,    0, "left")),
        ("bsarec",   "lbsp_mean",     "BSARec+LBSP",  "#0072B2", "P", (8,   -1, "left")),
        ("tisasrec", "none",          "TiSASRec",     "#D55E00", "D", (-8,  -2, "right")),
    ]
    ax1.set_yscale("log")
    for bb, pm, lbl, col, mk, (dx, dy, ha) in ITEMS_A:
        r = c_idx[("sports", bb, pm)]
        x = float(r["inf_ms_per_user"])
        y = float(r["inf_peak_vram_mb"])
        ax1.plot(x, y, marker=mk, color=col, mec="white", mew=0.6, ms=5.5, ls="None", zorder=3)
        ax1.annotate(
            lbl, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va="center", fontsize=6.6, color="#222222",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#777777", shrinkB=3)
        )
    ax1.set_xlim(0.0385, 0.0585)
    ax1.set_ylim(45, 760)
    ax1.set_yticks([60, 100, 200, 400, 600])
    ax1.set_yticklabels(["60", "100", "200", "400", "600"])
    ax1.minorticks_off()
    ax1.set_xlabel("Inference Latency (ms / user)", fontsize=8.0)
    ax1.set_ylabel("Peak Inference VRAM (MiB, log)", fontsize=8.0)
    grid(ax1, "both")

    # Panel (b): Training epoch time on Sports (all newly trained models)
    ax2 = axes[1]
    ax2.set_title("(b) Sports Training Time per Epoch", loc="left", fontweight="bold", pad=5, fontsize=8.3)
    ITEMS_B = [
        ("gru4rec",   "none",          "GRU4Rec",       "#555555"),
        ("fmlprec",   "none",          "FMLP-Rec",      "#0072B2"),
        ("sasrec",    "absp_mean",     "SASRec+ABSP",   "#00895E"),
        ("sasrec",    "lbsp_max",      "SASRec+Max",    "#7B61A8"),
        ("sasrec",    "lbsp_attn",     "SASRec+Attn",   "#7B61A8"),
        ("sasrec",    "lbsp_deepsets", "SASRec+Deep",   "#7B61A8"),
        ("lsab",      "none",          "LSAB (TKDD)",   "#C0392B"),
        ("lrurec",    "none",          "LRURec",        "#00895E"),
        ("bsarec",    "lbsp_mean",     "BSARec+LBSP",   "#0072B2"),
        ("ticoserec", "lbsp_mean",     "TiCoSeRec+LBSP","#C0392B"),
        ("tisasrec",  "lbsp_mean",     "TiSASRec+LBSP", "#D55E00"),
    ]
    ys = np.arange(len(ITEMS_B))
    vals = [float(c_idx[("sports", bb, pm)]["mean_train_epoch_s"]) for bb, pm, _, _ in ITEMS_B]
    cols = [c for _, _, _, c in ITEMS_B]
    ax2.barh(ys, vals, height=0.68, color=cols, alpha=0.88, zorder=2)
    for y_i, v in zip(ys, vals):
        ax2.text(v + 0.5, y_i, f"{v:.1f}s", va="center", ha="left", fontsize=6.6, color="#222222")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([lbl for _, _, lbl, _ in ITEMS_B], fontsize=6.9)
    ax2.set_xlim(0, 39.5)
    ax2.set_xticks([0, 10, 20, 30])
    ax2.set_xlabel("Mean Training Time (s / epoch, RTX 4090)", fontsize=8.0)
    grid(ax2, "x")

    save_audited(fig, "Fig10")


def main():
    master = load_csv("master_summary.csv")
    contrasts = load_csv("paired_contrasts.csv")
    costs = load_csv("unified_cost_profile.csv")
    build_fig5(master)
    build_fig7(master, contrasts)
    build_fig8(master, contrasts)
    build_fig10(costs)
    print("ALL FOUR REV2 FIGURES BUILT AND AUDITED.")


if __name__ == "__main__":
    main()
