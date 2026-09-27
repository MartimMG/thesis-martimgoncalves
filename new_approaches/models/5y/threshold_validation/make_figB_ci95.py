# Regenerates Figure B with the filled band swapped from the 25-75th percentile
# (interquartile range) to the ci_alpha (2.5th / 97.5th percentile, 95%) interval
# across replicates.
#
# The main notebook (threshold_2.0_copy.ipynb, CONFIG["band_inner_pctiles"]) fills
# the IQR and leaves the 95% interval undrawn -- CONFIG's own comment explains why:
# the 95% band on EC is dominated by the OOB event count (median ~10 events at 5 y)
# rather than by stability of the derivation rule, and it spans [0, 1] over the
# first two years for that reason. This script draws the wider band anyway, on
# request, so the two conventions can be compared side by side. No number changes:
# operating_characteristics.csv already carries both _q1/_q3 and _lo/_hi and this
# reads only the latter pair.
#
# Reads only operating_characteristics.csv + run_metadata.json, so it does not
# require re-running the 500-replicate bootstrap. Run from this directory.

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

META = json.load(open("run_metadata.json"))
C = META["config"]
oc = pd.read_csv("operating_characteristics.csv")

ALPHA = float(C["ci_alpha"])
OUTER_LABEL = f"{100 * (1 - ALPHA):.0f}%"

DIV, FS = C["time_display_divisor"], C["font_size"]
EV = oc["t"].to_numpy(float)
yrs = EV / DIV
prim = float(C["primary_horizon"]) / DIV
PJ = int(np.argmin(np.abs(EV - C["primary_horizon"])))
cap_years = float(C["administrative_cap_days"]) / DIV
ecc, fbc = C["palette_metrics"]
den = oc["event_capture_oob_denominator_median"].to_numpy(float)

plt.rcParams.update({"font.size": FS, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(C["figure_width_in"], 4.1), constrained_layout=True)

ax.axvline(prim, color="0.55", lw=0.9, ls=":", zorder=1)
ax.annotate("primary horizon (%.0f y)" % prim, xy=(prim, 1.0), xytext=(-4, -2),
            textcoords="offset points", ha="right", va="top", fontsize=FS - 2, color="0.35")

T = C["design_targets"]
for value, colour, text in (
    (T["min_event_capture"], ecc, r"target EC $\geq$ %.0f%%" % (100 * T["min_event_capture"])),
    (T["max_flag_burden"], fbc, r"target FB $\leq$ %.0f%%" % (100 * T["max_flag_burden"])),
):
    ax.axhline(value, color=colour, lw=0.9, ls=(0, (4, 3)), alpha=0.7, zorder=1)
    ax.annotate(text, xy=(0.02, value), xytext=(0, 3), textcoords="offset points",
                fontsize=FS - 2, color=colour, va="bottom")


def draw(prefix, colour):
    """Filled 95% band, centre line, apparent line over the whole grid."""
    lo = oc["%s_oob_lo" % prefix].to_numpy(float)
    hi = oc["%s_oob_hi" % prefix].to_numpy(float)
    ax.fill_between(yrs, lo, hi, color=colour, alpha=0.22, lw=0, zorder=2)
    ax.plot(yrs, oc["%s_oob_centre" % prefix].to_numpy(float), color=colour, zorder=4)
    ax.plot(yrs, oc["%s_apparent" % prefix].to_numpy(float), color=colour,
            ls="--", lw=1.2, zorder=4)


draw("flag_burden", fbc)
draw("event_capture", ecc)

for prefix, colour, tag in (("event_capture", ecc, "EC"), ("flag_burden", fbc, "FB")):
    v = float(oc["%s_oob_centre" % prefix].iloc[PJ])
    ax.annotate("%s %.0f%%" % (tag, 100 * v), xy=(yrs[PJ], v), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=FS - 1,
                color=colour, fontweight="bold")

ax.set_xlabel("Follow-up (%s)" % C["time_display_unit"])
ax.set_ylabel("Proportion")
ax.set_xlim(0, cap_years * 1.06)
ax.set_ylim(0, 1.0)

sec = ax.secondary_xaxis("top")
half = 0.5 * DIV
tick_idx, tick_pos = [], []
for yv in range(1, int(np.floor(EV.max() / DIV + 0.5)) + 1):
    j = int(np.argmin(np.abs(EV - yv * DIV)))
    if abs(EV[j] - yv * DIV) <= half:
        tick_idx.append(j)
        tick_pos.append(float(yrs[j]))
sec.set_xticks(tick_pos)
sec.set_xticklabels(["%.0f" % den[j] for j in tick_idx])
sec.set_xlabel("median OOB events available to EC($t$)", labelpad=2,
               fontsize=FS - 2, color="0.35")
sec.tick_params(labelsize=FS - 2, colors="0.35", length=2, pad=1)
sec.spines["top"].set_color("0.8")

handles = [
    Line2D([], [], color=ecc, lw=1.6, label="Event capture EC($t$), out-of-bag median"),
    Line2D([], [], color=ecc, lw=1.2, ls="--", label="Event capture, apparent"),
    Patch(facecolor="0.45", alpha=0.28, lw=0, label="%s CI of replicates" % OUTER_LABEL),
    Line2D([], [], color=fbc, lw=1.6, label="Flag burden FB($t$), out-of-bag median"),
    Line2D([], [], color=fbc, lw=1.2, ls="--", label="Flag burden, apparent"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
          ncol=2, frameon=False)

for ext in ("pdf", "png"):
    fig.savefig("figB_operating_characteristics_ci95.%s" % ext, dpi=200)
print("wrote figB_operating_characteristics_ci95.{pdf,png}")
