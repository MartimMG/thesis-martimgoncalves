# Regenerates Figure B with the event-capture interquartile band suppressed over
# the region where the band is an artefact of the estimator rather than a
# statement about performance.
#
# Why: before ~3 y the median replicate has <=5 out-of-bag recurrences, so EC(t)
# is a ratio of small integers whose distribution has a point mass of 72-99% at
# exactly 1.0. The 25th percentile of such a distribution sits at 1.0 while that
# mass exceeds 75% and falls off a cliff the moment it dips below. The mass
# crosses 75% at 2.18 -> 2.22 y (75.1% -> 74.0%), which is the step the eye reads
# as an anomaly, and again near 1 y when replicates whose first OOB event has
# just occurred enter the pool. Neither is a change in performance.
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

# Draw the band only where at least this many OOB events feed EC(t).
# NB at 4 the band starts at 720 d (1.97 y), which is BEFORE the 2.18 -> 2.22 y
# crossing, so the 1.00 -> 0.875 step in the lower quartile is still drawn. Six
# is the first count at which the point mass at EC = 1 has dissolved (74% -> 43%)
# and is the value that removes that step as well.
EC_BAND_MIN_EVENTS = 4

META = json.load(open("run_metadata.json"))
C = META["config"]
oc = pd.read_csv("operating_characteristics.csv")

DIV, FS = C["time_display_divisor"], C["font_size"]
EV = oc["t"].to_numpy(float)
yrs = EV / DIV
prim = float(C["primary_horizon"]) / DIV
PJ = int(np.argmin(np.abs(EV - C["primary_horizon"])))
cap_years = float(C["administrative_cap_days"]) / DIV
ecc, fbc = C["palette_metrics"]
den = oc["event_capture_oob_denominator_median"].to_numpy(float)
FIRST = int(np.argmax(den >= EC_BAND_MIN_EVENTS))
print("EC band starts at t = %.0f d (%.2f y), median denominator = %.0f events"
      % (EV[FIRST], yrs[FIRST], den[FIRST]))

plt.rcParams.update({"font.size": FS, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(C["figure_width_in"], 4.1), constrained_layout=True)

ax.axvline(prim, color="0.55", lw=0.9, ls=":", zorder=1)
ax.annotate("primary horizon (%.0f y)" % prim, xy=(prim, 1.0), xytext=(-4, -2),
            textcoords="offset points", ha="right", va="top", fontsize=FS - 2, color="0.35")

T = C["design_targets"]
ax.annotate(r"target FB $\approx$ %.0f%%" % (100 * T["max_flag_burden"]),
            xy=(0.02, T["max_flag_burden"]), xytext=(0, 3), textcoords="offset points",
            fontsize=FS - 2, color=fbc, va="bottom")


def draw(prefix, colour, band_slice):
    """Centre and apparent lines over the whole grid; band only over band_slice."""
    q1 = oc["%s_oob_q1" % prefix].to_numpy(float)
    q3 = oc["%s_oob_q3" % prefix].to_numpy(float)
    ax.fill_between(yrs[band_slice], q1[band_slice], q3[band_slice],
                    color=colour, alpha=0.22, lw=0, zorder=2)
    ax.plot(yrs, oc["%s_oob_centre" % prefix].to_numpy(float), color=colour, zorder=4)
    ax.plot(yrs, oc["%s_apparent" % prefix].to_numpy(float), color=colour,
            ls="--", lw=1.2, zorder=4)


draw("flag_burden", fbc, slice(None))
draw("event_capture", ecc, slice(FIRST, None))

# The un-banded stretch carries no in-figure marking: it is explained in the
# caption instead, so the plot area stays clean.

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
    Patch(facecolor="0.45", alpha=0.28, lw=0, label="25-75th pct of replicates"),
    Line2D([], [], color=fbc, lw=1.6, label="Flag burden FB($t$), out-of-bag median"),
    Line2D([], [], color=fbc, lw=1.2, ls="--", label="Flag burden, apparent"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
          ncol=2, frameon=False)

for ext in ("pdf", "png"):
    fig.savefig("figB_operating_characteristics_ecband.%s" % ext, dpi=200)
print("wrote figB_operating_characteristics_ecband.{pdf,png}")
