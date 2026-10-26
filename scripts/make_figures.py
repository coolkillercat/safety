"""Regenerate the paper's figures from the released result tables (terminal-answer rule, `_final` tables).

  python scripts/make_figures.py          (from the repository root; needs matplotlib + pandas)

Inputs  results/round2/crossover_placement_contrasts_final.csv   template A placement contrasts (known items)
        results/round2/crossover_C_contrasts_final.csv           template C placement contrasts (known items)
        results/round2/attribution_fouram_final.csv              package effects (boundary, initially correct)
Outputs figures/overview.{pdf,png}             (Figure 1; also reads crossover_cells_final.csv and
                                                             attribution_qwen3.5-27b_final.csv)
        figures/placement_contrasts.{pdf,png}   (Figure 3)
        figures/replay_packages.{pdf,png}       (Figure 4)
Per-cell error rates are in the appendix tables of the paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "round2"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# one colour per kind of advice, fixed across figures (validated for colour-vision deficiency)
BARE = {"color": "#1565C0", "marker": "o", "label": "bare note"}
RATIONALE = {"color": "#D84315", "marker": "s", "label": "note + false rationale"}
INK_MUTED = "#6B6B6B"

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9A9A9A", "xtick.color": "#9A9A9A",
                     "ytick.color": "#9A9A9A", "xtick.labelcolor": "#222222", "ytick.labelcolor": "#222222"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})  # no timestamp, so reruns are byte-identical
    fig.savefig(OUT / f"{name}.png", dpi=200)
    print("->", OUT / f"{name}.pdf")


def _norm(s: pd.Series) -> pd.Series:
    return s.str.replace("−", "-", regex=False)


def placement_contrasts(place_a: pd.DataFrame, place_c: pd.DataFrame):
    """Figure 3: placement effects under the bare note and under the false rationale, templates A and C."""
    place_a = place_a.assign(contrast=_norm(place_a.contrast), family=_norm(place_a.family))
    place_c = place_c.assign(contrast=_norm(place_c.contrast), family=_norm(place_c.family))
    rows = [("system message\n\u2212 after options", "system - user"),
            ("before question\n\u2212 after options", "userpre - user")]
    A = place_a[place_a.family == "organiser/peer only"]
    Ai = place_a[place_a.family == "tier interaction, organiser/peer (decept - none)"]
    C = place_c[(place_c.family == "placement main effect within advice") & (place_c.stratum == "known")]
    Ci = place_c[(place_c.family == "placement x rationale (decept - bare)".replace("x", "×")) & (place_c.stratum == "known")]

    def one(df, mask):
        r = df[mask]
        assert len(r) == 1, (len(r), mask)
        r = r.iloc[0]
        return float(r["diff"]), float(r["lo"]), float(r["hi"])

    panels = [
        ("original wording (template A)",
         lambda key: one(A, (A.tier == "none") & (A.contrast == f"placement {key}")),
         lambda key: one(A, (A.tier == "decept") & (A.contrast == f"placement {key}")),
         lambda key: one(Ai, Ai.contrast == f"placement {key}")),
        ("independent wording (template C)",
         lambda key: one(C, C.contrast == f"bare: {key}"),
         lambda key: one(C, C.contrast == f"decept: {key}"),
         lambda key: one(Ci, Ci.contrast == f"({key}) under decept - under bare")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), sharex=True, sharey=True)
    y = np.array([1.0, 0.0])
    for ax, (title, bare, rat, inter) in zip(axes, panels):
        for k, (label, key) in enumerate(rows):
            for spec, off in ((BARE, 0.16), (RATIONALE, -0.16)):
                m, lo, hi = (bare if spec is BARE else rat)(key)
                ax.errorbar([m], [y[k] + off], xerr=[[m - lo], [hi - m]], fmt=spec["marker"], ms=5,
                            color=spec["color"], ecolor=spec["color"], elinewidth=1.2, capsize=2.5,
                            label=spec["label"] if (k == 0 and ax is axes[0]) else None)
            d, dlo, dhi = inter(key)
            ax.text(64, y[k], f"{d:+.1f} ({dlo:.1f} to {dhi:.1f})", ha="right", va="center", fontsize=7.5,
                    color=INK_MUTED)
        ax.text(64, 1.72, "argument's extra placement effect", ha="right", va="center", fontsize=7, color=INK_MUTED,
                style="italic")
        ax.axvline(0, color="#9A9A9A", lw=1)
        ax.set_title(title)
        ax.set_xlim(-18, 66)
        ax.set_ylim(-0.6, 1.95)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows])
    fig.subplots_adjust(left=0.155, right=0.99, top=0.87, bottom=0.34, wspace=0.07)
    fig.supxlabel("placement effect (points): error in the earlier placement minus error after the options", fontsize=8.5, y=0.15)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, "placement_contrasts")


def replay_packages(fouram: pd.DataFrame):
    """Figure 4: package effect of each prior package (boundary items, initially correct)."""
    order = ["own-full", "quoted-full", "own-conclusion", "quoted-conclusion", "own-wrong-full", "filler"]
    labels = ["full replay, own turn", "full replay, quoted", "conclusion only, own turn", "conclusion only, quoted",
              "wrong solution, own turn", "filler, own turn"]
    d = fouram[(fouram.stratum == "boundary") & (fouram.baseline == "correct")]
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    y = np.arange(len(order))[::-1]
    for adv, spec, off in (("bare_wrong", BARE, 0.16), ("rationale_wrong", RATIONALE, -0.16)):
        m = [d[(d.prior == p) & (d.advice == adv)].fourarm_diff.iloc[0] for p in order]
        lo = [d[(d.prior == p) & (d.advice == adv)].lo.iloc[0] for p in order]
        hi = [d[(d.prior == p) & (d.advice == adv)].hi.iloc[0] for p in order]
        ax.errorbar(m, y + off, xerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt=spec["marker"], ms=5,
                    color=spec["color"], ecolor=spec["color"], capsize=2.5, elinewidth=1.2, label=spec["label"])
    ax.axvline(0, color="#9A9A9A", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("package effect (points); negative = the package protects the answer")
    ax.set_xlim(-38, 15)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower left", frameon=False, borderaxespad=0.3)
    fig.tight_layout()
    save(fig, "replay_packages")


def overview(cells: pd.DataFrame, attr: pd.DataFrame):
    """Figure 1: the three questions and the main result of each, as error rates (levels, no contrasts)."""
    A = cells[(cells.template == "A") & (cells.stratum == "known")]
    noadv = float(A[A.role == "noadv"].error.iloc[0])

    def mean_a(**kw):
        d = A
        for k, v in kw.items():
            d = d[d[k].isin(v) if isinstance(v, (list, tuple)) else d[k] == v]
        assert len(d) > 0, kw
        return float(d.error.mean())

    places = [("system", "system\nmessage"), ("userpre", "before\nquestion"), ("user", "after\noptions")]
    speakers = [("peer", "another\nparticipant"), ("expert", "named\nprofessor"), ("organiser", "session\norganiser")]
    a = {t: [mean_a(role=r, tier=t, speaker=["organiser", "peer"]) for r, _ in places] for t in ("none", "decept")}
    b = {t: [mean_a(speaker=sp, tier=t) for sp, _ in speakers] for t in ("none", "decept")}

    priors = [("none", "no prior\nturn"), ("own-full", "own\nreplay"), ("quoted-full", "quoted\nreplay")]
    B = attr[attr.stratum == "boundary"]

    def err(base, prior, adv):
        r = B[(B.baseline == base) & (B.prior == prior) & (B.advice == adv)]
        assert len(r) == 1, (base, prior, adv)
        return float(r.err_item.iloc[0])

    c = {k: [err("correct", pr, adv) for pr, _ in priors] for k, adv in (("bare", "bare_wrong"), ("rat", "rationale_wrong"))}
    d = {k: [100 - err("wrong", pr, adv) for pr, _ in priors] for k, adv in (("bare", "bare_correct"), ("rat", "rationale_correct"))}

    panels = [
        ("(a)  Where does the advice sit?", "An argument does most harm before the question,\na bare note after the options.",
         [l for _, l in places], a["none"], a["decept"], "error on known items (%)", True, None),
        ("(b)  Who signs it?", "A named source raises error;\nwhich authority barely matters.",
         [l for _, l in speakers], b["none"], b["decept"], "error on known items (%)", True, None),
        ("(c)  What has the model already written?", "A replayed correct answer is protected,\nwhoever it is attributed to.",
         [l for _, l in priors], c["bare"], c["rat"], "correct answers lost (%)", False, "advice: a wrong answer"),
        ("(d)  ... and what it costs", "The same replay blocks corrections\nof a wrong first answer.",
         [l for _, l in priors], d["bare"], d["rat"], "wrong answers fixed (%)", False, "advice: the correct answer"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6))
    w = 0.36
    for ax, (q, concl, labels, v_bare, v_rat, ylab, show_noadv, note) in zip(axes.flat, panels):
        x = np.arange(len(labels))
        for vals, spec, off in ((v_bare, BARE, -w / 2), (v_rat, RATIONALE, w / 2)):
            bars = ax.bar(x + off, vals, width=w, color=spec["color"], edgecolor="white", linewidth=0.6)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color="#333333")
        if show_noadv:
            ax.axhline(noadv, color=INK_MUTED, lw=0.8, ls=(0, (2, 2)))
        if note:
            ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(-0.18, 1.24, q, transform=ax.transAxes, fontsize=7.8, color=INK_MUTED, va="bottom")
        ax.text(-0.18, 1.03, concl, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color="#222222", va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BARE["color"]), plt.Rectangle((0, 0), 1, 1, color=RATIONALE["color"])]
    fig.legend(handles, ["bare note", "note + rationale"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.855, bottom=0.115, hspace=0.78, wspace=0.34)
    save(fig, "overview")


if __name__ == "__main__":
    placement_contrasts(pd.read_csv(RES / "crossover_placement_contrasts_final.csv"),
                        pd.read_csv(RES / "crossover_C_contrasts_final.csv"))
    replay_packages(pd.read_csv(RES / "attribution_fouram_final.csv"))
    overview(pd.read_csv(RES / "crossover_cells_final.csv"), pd.read_csv(RES / "attribution_qwen3.5-27b_final.csv"))
