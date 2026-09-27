from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sksurv.nonparametric import kaplan_meier_estimator


@dataclass
class DynamicThresholdResult:
	"""Container for dynamic threshold outputs."""

	thresholds: pd.DataFrame
	risk_scores_train: pd.DataFrame
	fitted_model: object
	# free-form bag for derivation diagnostics (flagged-fraction EB curves, prior, alpha, ...).
	# defaulted so old callers that don't know about it still work.
	diagnostics: Dict[str, object] = field(default_factory=dict)


def _extract_event_indicator(y: np.ndarray) -> np.ndarray:
	"""Pull the event indicator out of a sksurv structured array."""
	if getattr(y, "dtype", None) is None or y.dtype.names is None:
		raise ValueError("y must be a structured array with event/time fields.")

	# sksurv Surv arrays are always (event, time) in that field order.
	event_field = y.dtype.names[0]
	return y[event_field].astype(bool)


def _build_risk_matrix(
	model: object,
	x_data: pd.DataFrame,
	time_points: Iterable[float],
) -> np.ndarray:
	"""Build the (n_samples, n_time_points) risk matrix, risk = 1 - S(t)."""
	time_points = np.asarray(list(time_points), dtype=float)
	surv_funcs = model.predict_survival_function(x_data)

	risk_matrix = np.zeros((x_data.shape[0], time_points.shape[0]), dtype=float)
	for row_idx, fn in enumerate(surv_funcs):
		# clip query times to the function's own support, otherwise we get NaN/out-of-domain
		if hasattr(fn, "x") and np.size(fn.x) > 1:
			t_eval = np.clip(time_points, float(fn.x[0]), float(fn.x[-1]))
		else:
			t_eval = time_points

		# clip to [0, 1] just in case of float noise near the edges
		risk_values = 1.0 - fn(t_eval)
		risk_values = np.asarray(risk_values, dtype=float)
		risk_values = np.nan_to_num(risk_values, nan=0.0, posinf=1.0, neginf=0.0)
		risk_matrix[row_idx, :] = np.clip(risk_values, 0.0, 1.0)

	return risk_matrix


def _compute_m_from_costs(
	event_prevalence: float,
	false_positive_cost: float,
	false_negative_cost: float,
	true_positive_cost: float = 0.0,
	true_negative_cost: float = 0.0,
) -> float:
	"""Target ROC slope m, from prevalence and the confusion-matrix costs."""
	if not 0 < event_prevalence < 1:
		raise ValueError("event_prevalence must be in (0, 1).")
	numerator = float(false_positive_cost) - float(true_negative_cost)
	denominator = float(false_negative_cost) - float(true_positive_cost)
	if denominator <= 0:
		raise ValueError("(C_FN - C_TP) must be strictly positive.")
	if numerator < 0:
		raise ValueError("(C_FP - C_TN) must be non-negative.")

	healthy_prevalence = 1.0 - event_prevalence
	m_value = (healthy_prevalence / event_prevalence) * (numerator / denominator)
	return float(m_value)


def _extract_time_values(y: np.ndarray) -> np.ndarray:
	"""Companion to _extract_event_indicator: pulls out the time field instead."""
	if getattr(y, "dtype", None) is None or y.dtype.names is None:
		raise ValueError("y must be a structured array with event/time fields.")
	time_field = y.dtype.names[1]
	return np.asarray(y[time_field], dtype=float)


def _km_survival_at_times(
	events: np.ndarray,
	times: np.ndarray,
	time_points: np.ndarray,
) -> np.ndarray:
	"""Kaplan-Meier S(t) evaluated at arbitrary query times."""
	km_times, km_survival = kaplan_meier_estimator(
		np.asarray(events, dtype=bool),
		np.asarray(times, dtype=float),
	)

	# right-continuous step function: for each t, take the latest KM value with time <= t
	km_times = np.asarray(km_times, dtype=float)
	km_survival = np.asarray(km_survival, dtype=float)
	time_points = np.asarray(time_points, dtype=float)

	survival_at_t = np.ones_like(time_points, dtype=float)
	indices = np.searchsorted(km_times, time_points, side="right") - 1
	valid = indices >= 0
	survival_at_t[valid] = km_survival[indices[valid]]
	return np.clip(survival_at_t, 0.0, 1.0)


def _step_survival_at_times(
	step_times: np.ndarray,
	step_survival: np.ndarray,
	query_times: np.ndarray,
) -> np.ndarray:
	"""Same step-function lookup as _km_survival_at_times, but for an already-built curve."""
	step_times = np.asarray(step_times, dtype=float)
	step_survival = np.asarray(step_survival, dtype=float)
	query_times = np.asarray(query_times, dtype=float)

	out = np.ones_like(query_times, dtype=float)
	indices = np.searchsorted(step_times, query_times, side="right") - 1
	valid = indices >= 0
	out[valid] = step_survival[indices[valid]]
	return np.clip(out, 0.0, 1.0)


def _build_bootstrap_strata_labels(
	events: np.ndarray,
	times: np.ndarray,
	max_time_bins: int,
) -> np.ndarray:
	"""Stratum label = event indicator + a binned time quantile, for the bootstrap below."""
	events_int = np.asarray(events, dtype=bool).astype(int)
	times_float = np.asarray(times, dtype=float)

	if max_time_bins < 2:
		return events_int.astype(str)

	# try decreasing numbers of qcut bins until pd.qcut actually gives us >=2 distinct
	# ones (it silently collapses bins on tied/sparse data, hence duplicates="drop")
	chosen_time_bins = None
	for n_bins in range(int(max_time_bins), 1, -1):
		try:
			time_bins = pd.qcut(times_float, q=n_bins, labels=False, duplicates="drop")
			time_bins = pd.Series(time_bins)
			if int(time_bins.nunique()) >= 2:
				chosen_time_bins = time_bins.astype(int)
				break
		except ValueError:
			continue

	if chosen_time_bins is None:
		# too sparse for any time binning, just stratify on event status
		return events_int.astype(str)

	return events_int.astype(str) + "_" + chosen_time_bins.astype(str)


def _stratified_bootstrap_indices(
	rng: np.random.Generator,
	strata_labels: np.ndarray,
) -> np.ndarray:
	"""Bootstrap resample, but within each stratum separately so stratum sizes don't drift."""
	strata_labels = np.asarray(strata_labels)
	if strata_labels.ndim != 1:
		raise ValueError("strata_labels must be a 1D array.")

	sampled_parts = []
	for label in np.unique(strata_labels):
		stratum_idx = np.where(strata_labels == label)[0]
		if stratum_idx.size == 0:
			continue
		sampled = rng.choice(stratum_idx, size=stratum_idx.size, replace=True)
		sampled_parts.append(sampled)

	if not sampled_parts:
		raise ValueError("No samples available to bootstrap from the provided strata.")

	sample_idx = np.concatenate(sampled_parts)
	rng.shuffle(sample_idx)
	return sample_idx


def _compute_threshold_from_roc_utility(
	scores: np.ndarray,
	events: np.ndarray,
	times: np.ndarray,
	time_point: float,
	m_value: float,
	sample_weights: Optional[np.ndarray] = None,
) -> Tuple[float, int, float]:
	"""Threshold that maximizes U(c) = TPR(c) - m*FPR(c) at a given horizon.

	Positives are events that happened by time_point; anyone censored before that
	is dropped since we don't know their status yet.
	"""
	# need a known outcome by t: either an event on or before t, or still under
	# observation at t (censored after t tells us nothing either way)
	valid_mask = (times >= time_point) | (events & (times <= time_point))
	if valid_mask.sum() < 3:
		return float(np.nan), int(valid_mask.sum()), float(np.nan)

	y_bin = (events[valid_mask] & (times[valid_mask] <= time_point)).astype(int)
	x_scores = scores[valid_mask]

	# ROC is undefined with only one class present
	if y_bin.min() == y_bin.max():
		return float(np.nan), int(valid_mask.sum()), float(np.nan)

	if sample_weights is not None:
		w_all = np.asarray(sample_weights, dtype=float)
		if w_all.shape[0] != scores.shape[0]:
			raise ValueError("sample_weights must have the same length as scores.")
		w = w_all[valid_mask]
		w = np.clip(w, 0.0, np.inf)

		# same issue as above but for total weight rather than raw counts
		w_pos = float(np.sum(w[y_bin == 1]))
		w_neg = float(np.sum(w[y_bin == 0]))
		if w_pos <= 0.0 or w_neg <= 0.0:
			return float(np.nan), int(valid_mask.sum()), float(np.nan)

		fpr, tpr, thresholds = roc_curve(
			y_bin,
			x_scores,
			sample_weight=w,
			drop_intermediate=False,
		)
	else:
		fpr, tpr, thresholds = roc_curve(y_bin, x_scores, drop_intermediate=False)
	utilities = tpr - (m_value * fpr)
	best_idx = int(np.argmax(utilities))
	roc_threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))
	max_utility = float(utilities[best_idx])

	return roc_threshold, int(valid_mask.sum()), max_utility


def _compute_threshold_from_roc_youden(
	scores: np.ndarray,
	events: np.ndarray,
	times: np.ndarray,
	time_point: float,
	sample_weights: Optional[np.ndarray] = None,
) -> Tuple[float, int, float]:
	"""Same construction as _compute_threshold_from_roc_utility, but plain Youden's J
	(TPR - FPR) instead of the cost-weighted utility -- used for t_intermediate."""
	# same eligibility rule as the utility version above
	valid_mask = (times >= time_point) | (events & (times <= time_point))
	if valid_mask.sum() < 3:
		return float(np.nan), int(valid_mask.sum()), float(np.nan)

	y_bin = (events[valid_mask] & (times[valid_mask] <= time_point)).astype(int)
	x_scores = scores[valid_mask]

	if y_bin.min() == y_bin.max():
		return float(np.nan), int(valid_mask.sum()), float(np.nan)

	if sample_weights is not None:
		w_all = np.asarray(sample_weights, dtype=float)
		if w_all.shape[0] != scores.shape[0]:
			raise ValueError("sample_weights must have the same length as scores.")
		w = w_all[valid_mask]
		w = np.clip(w, 0.0, np.inf)

		w_pos = float(np.sum(w[y_bin == 1]))
		w_neg = float(np.sum(w[y_bin == 0]))
		if w_pos <= 0.0 or w_neg <= 0.0:
			return float(np.nan), int(valid_mask.sum()), float(np.nan)

		fpr, tpr, thresholds = roc_curve(
			y_bin,
			x_scores,
			sample_weight=w,
			drop_intermediate=False,
		)
	else:
		fpr, tpr, thresholds = roc_curve(y_bin, x_scores, drop_intermediate=False)

	youden_j = tpr - fpr
	best_idx = int(np.argmax(youden_j))
	youden_threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))
	max_j = float(youden_j[best_idx])

	return youden_threshold, int(valid_mask.sum()), max_j


def _natural_cubic_spline_predict(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_query: np.ndarray,
) -> np.ndarray:
	"""Natural cubic spline (second derivative = 0 at the boundary knots), extrapolated linearly."""
	x_train = np.asarray(x_train, dtype=float)
	y_train = np.asarray(y_train, dtype=float)
	x_query = np.asarray(x_query, dtype=float)

	if x_train.ndim != 1 or y_train.ndim != 1 or x_train.shape[0] != y_train.shape[0]:
		raise ValueError("x_train and y_train must be 1D arrays of equal length.")
	if x_train.shape[0] == 0:
		raise ValueError("At least one training point is required for spline fitting.")

	order = np.argsort(x_train)
	x_sorted = x_train[order]
	y_sorted = y_train[order]

	# duplicate x's break the tridiagonal system below, so average their y's first
	unique_x, inverse = np.unique(x_sorted, return_inverse=True)
	if unique_x.shape[0] != x_sorted.shape[0]:
		sums = np.zeros(unique_x.shape[0], dtype=float)
		counts = np.zeros(unique_x.shape[0], dtype=float)
		np.add.at(sums, inverse, y_sorted)
		np.add.at(counts, inverse, 1.0)
		x_sorted = unique_x
		y_sorted = sums / counts

	n = x_sorted.shape[0]
	if n == 1:
		return np.full_like(x_query, y_sorted[0], dtype=float)
	if n == 2:
		return np.interp(x_query, x_sorted, y_sorted)

	h = np.diff(x_sorted)
	if np.any(h <= 0):
		raise ValueError("x_train must contain strictly increasing values.")

	a = np.zeros((n, n), dtype=float)
	b = np.zeros(n, dtype=float)
	a[0, 0] = 1.0
	a[-1, -1] = 1.0

	for i in range(1, n - 1):
		a[i, i - 1] = h[i - 1]
		a[i, i] = 2.0 * (h[i - 1] + h[i])
		a[i, i + 1] = h[i]
		b[i] = 6.0 * (
			(y_sorted[i + 1] - y_sorted[i]) / h[i]
			- (y_sorted[i] - y_sorted[i - 1]) / h[i - 1]
		)

	m = np.linalg.solve(a, b)

	# slopes at the two boundary knots, used to extrapolate linearly beyond the data
	slope_left = (y_sorted[1] - y_sorted[0]) / h[0] - (h[0] / 6.0) * (2.0 * m[0] + m[1])
	h_last = h[-1]
	slope_right = (y_sorted[-1] - y_sorted[-2]) / h_last + (h_last / 6.0) * (2.0 * m[-1] + m[-2])

	y_query = np.empty_like(x_query, dtype=float)
	left_mask = x_query < x_sorted[0]
	right_mask = x_query > x_sorted[-1]
	inside_mask = ~(left_mask | right_mask)

	if np.any(left_mask):
		y_query[left_mask] = y_sorted[0] + slope_left * (x_query[left_mask] - x_sorted[0])
	if np.any(right_mask):
		y_query[right_mask] = y_sorted[-1] + slope_right * (x_query[right_mask] - x_sorted[-1])

	if np.any(inside_mask):
		xi = x_query[inside_mask]
		idx = np.searchsorted(x_sorted, xi, side="right") - 1
		idx = np.clip(idx, 0, n - 2)

		x_l = x_sorted[idx]
		x_r = x_sorted[idx + 1]
		h_i = x_r - x_l
		a_i = (x_r - xi) / h_i
		b_i = (xi - x_l) / h_i

		y_i = (
			a_i * y_sorted[idx]
			+ b_i * y_sorted[idx + 1]
			+ (((a_i ** 3) - a_i) * m[idx] + ((b_i ** 3) - b_i) * m[idx + 1]) * (h_i ** 2) / 6.0
		)
		y_query[inside_mask] = y_i

	return y_query


def _cubic_spline_predict(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_query: np.ndarray,
) -> np.ndarray:
	"""Standard cubic spline (not-a-knot end condition via SciPy, if it's installed)."""
	x_train = np.asarray(x_train, dtype=float)
	y_train = np.asarray(y_train, dtype=float)
	x_query = np.asarray(x_query, dtype=float)

	if x_train.ndim != 1 or y_train.ndim != 1 or x_train.shape[0] != y_train.shape[0]:
		raise ValueError("x_train and y_train must be 1D arrays of equal length.")
	if x_train.shape[0] == 0:
		raise ValueError("At least one training point is required for spline fitting.")

	order = np.argsort(x_train)
	x_sorted = x_train[order]
	y_sorted = y_train[order]

	# same duplicate-x handling as the natural spline above
	unique_x, inverse = np.unique(x_sorted, return_inverse=True)
	if unique_x.shape[0] != x_sorted.shape[0]:
		sums = np.zeros(unique_x.shape[0], dtype=float)
		counts = np.zeros(unique_x.shape[0], dtype=float)
		np.add.at(sums, inverse, y_sorted)
		np.add.at(counts, inverse, 1.0)
		x_sorted = unique_x
		y_sorted = sums / counts

	n = x_sorted.shape[0]
	if n == 1:
		return np.full_like(x_query, y_sorted[0], dtype=float)
	if n == 2:
		return np.interp(x_query, x_sorted, y_sorted)

	try:
		from scipy.interpolate import CubicSpline

		# not-a-knot is SciPy's default bc_type, which is what "standard cubic spline" means here
		cs = CubicSpline(x_sorted, y_sorted, bc_type="not-a-knot", extrapolate=True)
		return np.asarray(cs(x_query), dtype=float)
	except Exception:
		# no scipy? fall back to our own natural-spline implementation
		return _natural_cubic_spline_predict(x_sorted, y_sorted, x_query)


def _compute_high_thresholds_with_rcs(
	time_points: np.ndarray,
	event_times: np.ndarray,
	high_curve_values: np.ndarray,
	high_threshold_anchor_points: Optional[Dict[float, float]],
	spline_quantiles: Tuple[float, ...],
	spline_variant: str = "restricted",
) -> np.ndarray:
	"""Smooth t_high by fitting a spline through knots placed at event-time quantiles.

	If anchor points are supplied they override the knot values; otherwise the knots
	are just the original ROC-utility curve sampled at those quantile times.
	"""

	if len(spline_quantiles) < 3:
		raise ValueError("high_threshold_spline_quantiles must contain at least 3 quantiles.")

	q_arr = np.asarray(spline_quantiles, dtype=float)
	if np.any((q_arr <= 0.0) | (q_arr >= 1.0)):
		raise ValueError("high_threshold_spline_quantiles values must be strictly between 0 and 1.")

	event_times = np.asarray(event_times, dtype=float)
	if event_times.size == 0:
		raise ValueError("Cannot fit RCS high thresholds because no event times were found.")

	time_points = np.asarray(time_points, dtype=float)
	high_curve_values = np.asarray(high_curve_values, dtype=float)
	if high_curve_values.shape[0] != time_points.shape[0]:
		raise ValueError("high_curve_values must have the same length as time_points.")
	high_curve_values = np.clip(high_curve_values, 0.0, 1.0)

	quantile_times = np.quantile(event_times, q_arr)
	quantile_times = np.asarray(quantile_times, dtype=float)
	unique_knot_times = np.unique(quantile_times)
	if unique_knot_times.shape[0] < 2:
		raise ValueError("Event-time quantiles produced fewer than two unique knot times.")

	if high_threshold_anchor_points is not None and len(high_threshold_anchor_points) > 0:
		anchor_times = np.asarray(sorted(high_threshold_anchor_points.keys()), dtype=float)
		anchor_values = np.asarray(
			[high_threshold_anchor_points[t] for t in sorted(high_threshold_anchor_points.keys())],
			dtype=float,
		)
		anchor_values = np.clip(anchor_values, 0.0, 1.0)
		if anchor_times.shape[0] < 2:
			raise ValueError("Provide at least two anchor points for high-threshold spline fitting.")
		knot_thresholds = np.interp(
			unique_knot_times,
			anchor_times,
			anchor_values,
			left=anchor_values[0],
			right=anchor_values[-1],
		)
	else:
		# no anchors given: just read the original curve off at the knot times
		knot_thresholds = np.interp(
			unique_knot_times,
			time_points,
			high_curve_values,
			left=high_curve_values[0],
			right=high_curve_values[-1],
		)

	if unique_knot_times.shape[0] < 3:
		# not enough distinct knots for a cubic fit, just connect the dots
		return np.clip(np.interp(time_points, unique_knot_times, knot_thresholds), 0.0, 1.0)

	if spline_variant == "restricted":
		spline_values = _natural_cubic_spline_predict(
			x_train=unique_knot_times,
			y_train=knot_thresholds,
			x_query=np.asarray(time_points, dtype=float),
		)
	elif spline_variant == "normal":
		spline_values = _cubic_spline_predict(
			x_train=unique_knot_times,
			y_train=knot_thresholds,
			x_query=np.asarray(time_points, dtype=float),
		)
	else:
		raise ValueError("spline_variant must be 'restricted' or 'normal'.")

	# flatten the tails: before the first observed event we just hold the value at
	# the first event time, and after the last event we hold the last-event value.
	# keeps the curve from doing anything weird where there's no event data to anchor it.
	time_points_arr = np.asarray(time_points, dtype=float)
	first_event_time = float(np.min(event_times))
	last_event_time = float(np.max(event_times))
	first_event_threshold = float(
		np.interp(first_event_time, time_points_arr, spline_values)
	)
	last_event_threshold = float(
		np.interp(last_event_time, time_points_arr, spline_values)
	)
	pre_event_mask = time_points_arr < first_event_time
	post_event_mask = time_points_arr > last_event_time
	if np.any(pre_event_mask):
		spline_values = np.asarray(spline_values, dtype=float)
		spline_values[pre_event_mask] = first_event_threshold
	if np.any(post_event_mask):
		spline_values = np.asarray(spline_values, dtype=float)
		spline_values[post_event_mask] = last_event_threshold

	return np.clip(spline_values, 0.0, 1.0)


def _rolling_mean_smooth_curve(
	curve_values: np.ndarray,
	window_days: int,
) -> np.ndarray:
	"""Centred rolling mean of a threshold curve, bounded by its own window.

	This is the alternative to `_smooth_curve_with_event_quantile_spline` at the
	same point in the pipeline. Less flexible, but it buys real guarantees:

	- the output at t is between the min and max of the input over t's own window,
	  so it can't undershoot the way a spline through event-time quantile knots
	  can. (Concretely: a natural cubic spline falling from a degenerate cutoff of
	  1.0 at its first knot to ~0.02 at its second dips to ~0.001 in between --
	  below every subject's risk -- which silently flags everybody over that
	  stretch. Rolling mean can't do that.)
	- no discontinuity can appear that wasn't already in the input.
	- nothing gets extrapolated: at the edges the window just shrinks instead of
	  continuing some boundary slope past the last knot.

	NaNs pass through untouched and don't count towards neighbours' means.
	"""
	curve = np.asarray(curve_values, dtype=float)
	window_days = int(window_days)
	if window_days < 1 or window_days % 2 == 0:
		raise ValueError("curve_rolling_mean_window must be a positive odd integer.")
	if window_days == 1 or curve.size < 3:
		return curve.copy()
	half = window_days // 2
	finite = np.isfinite(curve)
	filled = np.where(finite, curve, 0.0)
	cum = np.concatenate([[0.0], np.cumsum(filled)])
	cnt = np.concatenate([[0.0], np.cumsum(finite.astype(float))])
	idx = np.arange(curve.size)
	# shrink the window symmetrically at the edges rather than just clipping the side
	# that runs off the array. clipping only one side gives a ONE-SIDED average at the
	# edges, which biases a monotone curve towards the interior -- the last day of the
	# grid would average over the preceding half-window only, every value of it lower,
	# so e.g. a 20th-percentile curve ends up below the true 20th percentile exactly at
	# the horizon we report on. shrinking both sides keeps it centred: last day is its
	# own raw value, the day before is a 3-day mean, interior untouched.
	half_eff = np.minimum(half, np.minimum(idx, curve.size - 1 - idx))
	lo = idx - half_eff
	hi = idx + half_eff + 1
	n_used = cnt[hi] - cnt[lo]
	out = np.divide(cum[hi] - cum[lo], n_used,
					out=np.full(curve.shape, np.nan), where=n_used > 0)
	return np.clip(np.where(finite, out, curve), 0.0, 1.0)


def _smooth_curve_with_event_quantile_spline(
	time_points: np.ndarray,
	event_times: np.ndarray,
	curve_values: np.ndarray,
	spline_quantiles: Tuple[float, ...],
	spline_variant: str = "restricted",
	min_knot_time: Optional[float] = None,
	zero_start_time: Optional[float] = None,
	terminal_extrapolation: str = "linear",
) -> np.ndarray:
	"""Smooth a threshold curve over event-time quantile knots.

	Generic additive post-processing step, applied to a low/intermediate/high
	curve after it's already been computed elsewhere.

	A few knobs worth explaining:

	min_knot_time throws away any knot placed before that day. The idea is that
	the per-horizon ROC estimate isn't informative yet where too few events have
	accrued, so we shouldn't let the spline anchor to it. Surviving knots keep
	their original event-time quantile positions.

	zero_start_time pins the day the curve actually leaves zero. Left unset, the
	restricted spline just continues as a straight line below the first knot,
	which crosses zero wherever (first knot - value/slope) happens to land. With
	it set, that segment becomes a quadratic that's zero at zero_start_time and
	matches the spline's value and slope at the first knot -- so the fitted
	region is untouched and the join stays smooth.

	terminal_extrapolation controls what happens ABOVE the last knot. "linear"
	(the historical default) continues the spline's right boundary slope out to
	the last event time and then holds flat. "flat" just holds the last knot's
	value directly. Note the last knot sits at the upper event-time quantile, so
	this segment was never fitted on anything -- it's the boundary slope of a
	spline through the outer knot, extended over the remaining follow-up tail.
	When that slope is negative, "linear" drags the curve down over exactly the
	horizons with the most accrued events, i.e. where it matters most.
	"""
	time_points = np.asarray(time_points, dtype=float)
	event_times = np.asarray(event_times, dtype=float)
	curve_values = np.asarray(curve_values, dtype=float)

	if curve_values.shape[0] != time_points.shape[0]:
		raise ValueError("curve_values must have the same length as time_points.")
	if terminal_extrapolation not in {"linear", "flat"}:
		raise ValueError("terminal_extrapolation must be 'linear' or 'flat'.")
	curve_values = np.clip(curve_values, 0.0, 1.0)

	if event_times.size == 0:
		# nothing to anchor the knots to, leave the curve as-is
		return curve_values

	if len(spline_quantiles) < 3:
		raise ValueError("spline_quantiles must contain at least 3 quantiles.")

	q_arr = np.asarray(spline_quantiles, dtype=float)
	if np.any((q_arr <= 0.0) | (q_arr >= 1.0)):
		raise ValueError("spline_quantiles values must be strictly between 0 and 1.")

	quantile_times = np.quantile(event_times, q_arr)
	quantile_times = np.asarray(quantile_times, dtype=float)
	unique_knot_times = np.unique(quantile_times)

	if min_knot_time is not None:
		kept_knot_times = unique_knot_times[unique_knot_times >= float(min_knot_time)]
		if kept_knot_times.shape[0] < 2:
			raise ValueError(
				f"min_knot_time={float(min_knot_time)} leaves {kept_knot_times.shape[0]} knot(s) "
				f"out of {unique_knot_times.shape[0]}; lower it or widen spline_quantiles."
			)
		unique_knot_times = kept_knot_times

	if unique_knot_times.shape[0] < 2:
		return curve_values

	knot_values = np.interp(
		unique_knot_times,
		time_points,
		curve_values,
		left=curve_values[0],
		right=curve_values[-1],
	)

	if unique_knot_times.shape[0] < 3:
		spline_values = np.interp(time_points, unique_knot_times, knot_values)
	else:
		if spline_variant == "restricted":
			spline_values = _natural_cubic_spline_predict(
				x_train=unique_knot_times,
				y_train=knot_values,
				x_query=time_points,
			)
		elif spline_variant == "normal":
			spline_values = _cubic_spline_predict(
				x_train=unique_knot_times,
				y_train=knot_values,
				x_query=time_points,
			)
		else:
			raise ValueError("spline_variant must be 'restricted' or 'normal'.")

	# clamp behavior outside the observed event range so it stays sane
	spline_values = np.asarray(spline_values, dtype=float)
	first_event_time = float(np.min(event_times))
	last_event_time = float(np.max(event_times))

	if terminal_extrapolation == "flat":
		# hold at the last knot's value instead of continuing its boundary slope.
		# this subsumes the post-last-event flattening a few lines down (which then
		# just re-reads the same constant, i.e. a no-op) -- so the "linear" branch
		# below is untouched and stays byte-for-byte identical to before.
		last_knot_day = float(unique_knot_times[-1])
		last_knot_value = float(np.interp(last_knot_day, time_points, spline_values))
		beyond_last_knot = time_points > last_knot_day
		if np.any(beyond_last_knot):
			spline_values[beyond_last_knot] = last_knot_value

	last_event_value = float(np.interp(last_event_time, time_points, spline_values))

	post_event_mask = time_points > last_event_time
	if np.any(post_event_mask):
		spline_values[post_event_mask] = last_event_value

	if zero_start_time is None:
		first_event_value = float(np.interp(first_event_time, time_points, spline_values))
		pre_event_mask = time_points < first_event_time
		if np.any(pre_event_mask):
			spline_values[pre_event_mask] = first_event_value
	else:
		# swap the straight-line extrapolation below the first knot for the quadratic
		# through (zero_start_time, 0) matching the spline's value and slope at the
		# knot: f(t) = a*dt^2 + b*dt, dt = t - zero_start_time.
		zero_day = float(zero_start_time)
		knot_day = float(unique_knot_times[0])
		if zero_day >= knot_day:
			raise ValueError(
				f"zero_start_time={zero_day} must be before the first knot at day {knot_day:.0f}."
			)
		knot_value = float(np.interp(knot_day, time_points, spline_values))
		knot_slope = float(np.interp(knot_day + 1.0, time_points, spline_values) - knot_value)
		if knot_value <= 0.0 or knot_slope <= 0.0:
			raise ValueError(
				"zero_start_time needs a positive value and slope at the first knot "
				f"(got value={knot_value:.3e}, slope={knot_slope:.3e})."
			)
		span = knot_day - zero_day
		b_coef = 2.0 * knot_value / span - knot_slope
		if b_coef < 0.0:
			raise ValueError(
				f"zero_start_time={zero_day:.0f} is too early to join the first knot "
				f"monotonically; earliest is day {knot_day - 2.0 * knot_value / knot_slope:.0f}."
			)
		a_coef = (knot_slope * span - knot_value) / (span * span)
		below_mask = time_points < knot_day
		delta = time_points[below_mask] - zero_day
		spline_values[below_mask] = np.where(delta > 0.0, a_coef * delta ** 2 + b_coef * delta, 0.0)

	return np.clip(spline_values, 0.0, 1.0)


def _empirical_bayes_shrink_curve(
	base_curve: np.ndarray,
	support_counts: np.ndarray,
	shrink_k: float,
	direction: str,
) -> np.ndarray:
	"""Shrink a threshold curve toward a running-extremum prior, EB-style.

	shrunk_t = alpha_t * base_t + (1 - alpha_t) * prior_t, alpha_t = support_t / (support_t + k).
	Prior is the running max of the base curve for direction="high", running min for "low" -
	i.e. we pull low-support horizons towards whatever the curve has already committed to.
	"""
	base_arr = np.asarray(base_curve, dtype=float)
	support_arr = np.asarray(support_counts, dtype=float)

	if base_arr.ndim != 1 or support_arr.ndim != 1:
		raise ValueError("base_curve and support_counts must be 1D arrays.")
	if base_arr.shape[0] != support_arr.shape[0]:
		raise ValueError("base_curve and support_counts must have the same length.")
	if shrink_k <= 0.0:
		raise ValueError("shrink_k must be > 0.")

	base_arr = np.clip(base_arr, 0.0, 1.0)
	support_arr = np.clip(support_arr, 0.0, np.inf)

	if direction == "high":
		prior_arr = np.maximum.accumulate(base_arr)
	elif direction == "low":
		prior_arr = np.minimum.accumulate(base_arr)
	else:
		raise ValueError("direction must be 'high' or 'low'.")

	alpha = support_arr / (support_arr + float(shrink_k))
	shrunk = alpha * base_arr + (1.0 - alpha) * prior_arr
	return np.clip(shrunk, 0.0, 1.0)


def _cutoff_for_flagged_fraction(column: np.ndarray, target_fraction: float) -> float:
	"""Smallest risk cutoff c with #{i : r_i >= c} / n <= target_fraction.

	Why not just take a quantile? Because the risk column is heavily tied early
	in follow-up (day 1: almost everyone has r_i = 0), so the (1-q) quantile is
	also 0 and "r >= c" ends up flagging the whole cohort instead of the top q.
	Stepping to the smallest DISTINCT value under the cap avoids that: where the
	model can't separate patients it flags fewer of them, never all of them.
	"""
	column = np.asarray(column, dtype=float)
	n = int(column.size)
	if n == 0:
		return float("nan")

	top = float(np.max(column))
	max_flagged = int(np.floor(float(target_fraction) * n + 1e-9))
	if max_flagged <= 0:
		# target rounds down to zero people: sit just above the max observed risk
		return float(min(1.0, np.nextafter(top, np.inf)))

	ordered = np.sort(column)
	distinct = np.unique(ordered)
	count_at_or_above = n - np.searchsorted(ordered, distinct, side="left")
	admissible = np.flatnonzero(count_at_or_above <= max_flagged)
	if admissible.size == 0:
		# everyone ties at the same risk here, so no cutoff separates them -
		# flag nobody rather than everybody
		return float(min(1.0, np.nextafter(top, np.inf)))
	return float(distinct[admissible[0]])


def _empirical_bayes_shrink_flagged_fraction(
	risk_matrix: np.ndarray,
	high_curve: np.ndarray,
	cumulative_events: np.ndarray,
	shrink_k: float,
	min_events_for_prior: int,
	enforce_monotone: bool = True,
	prior_statistic: str = "median",
) -> Tuple[np.ndarray, Dict[str, object]]:
	"""Empirical-Bayes stabilisation of t_high, done in flagged-fraction (percentile) space.

	Why this exists: the per-horizon high threshold is the argmax of a ROC utility
	whose positives are the events accrued by that horizon. Early in follow-up that
	count is tiny (single digits), so the raw curve there is basically unidentified.
	Curve-shape patches applied directly to it (deleting early spline knots, pinning
	the day it leaves zero) end up tuned to one sample and don't survive resampling.

	So instead we borrow strength from horizons that do have events, via two choices
	that both differ from the legacy risk-space EB stage elsewhere in this file:

	- shrinkage weight uses events ACCRUED BY t, not events remaining after t. Accrued
	  events are what the ROC at t is actually built from, so that's where the data is
	  thin. (The old risk-space version used remaining-event count, which is largest
	  early and smallest late -- backwards from what we want.)
	- the prior is drawn from LATER horizons and applied to EARLIER ones. The old
	  running-maximum prior borrows from the left, so at the start of the grid the
	  prior just equals the early (unstable) value itself -- a no-op exactly where
	  shrinkage was supposed to help.

	Shrinkage happens on the flagged fraction, not the raw risk scale, because under
	proportional hazards the patient ranking is time-invariant: a constant flagged
	fraction flags the same people at every horizon, but a constant risk cutoff
	doesn't (1 - S(t) rises for everyone). That invariance is what lets one prior
	mean the same thing across the whole grid.

	Mechanically, with n_ev(t) = events accrued by t and n = fitting sample size:

		1.  q_raw(t) = (1/n) * #{i : r_i(t) >= c_high(t)}     -- flagged fraction
		2.  S = {t : n_ev(t) >= min_events_for_prior}         -- supported horizons
		    q_prior = median over S of q_raw (or its value at min S)
		3.  alpha(t) = n_ev(t) / (n_ev(t) + k)
		    q_hat(t) = alpha(t)*q_raw(t) + (1-alpha(t))*q_prior, optionally forced non-decreasing
		4.  c_high(t) = Quantile_{1 - q_hat(t)} of {r_i(t)}

	Where nothing has accrued yet, alpha(t) = 0 and the curve is exactly the prior --
	the early grid comes from borrowed information, not extrapolation.

	One more thing: forcing q_hat non-decreasing nests the flagged sets in t, so "ever
	flagged by t" and "flagged at t" coincide, and the flag burden at t is exactly q_hat(t).

	Returns (shrunk_curve, diagnostics_dict).
	"""
	risk_matrix = np.asarray(risk_matrix, dtype=float)
	high_curve = np.asarray(high_curve, dtype=float)
	cumulative_events = np.asarray(cumulative_events, dtype=float)

	if risk_matrix.ndim != 2:
		raise ValueError("risk_matrix must be 2D (n_samples, n_time_points).")
	n_samples, n_times = risk_matrix.shape
	if n_samples == 0:
		raise ValueError("risk_matrix must contain at least one sample.")
	if high_curve.shape[0] != n_times or cumulative_events.shape[0] != n_times:
		raise ValueError("high_curve and cumulative_events must match the time grid.")
	if shrink_k <= 0.0:
		raise ValueError("high_percentile_eb_k must be > 0.")
	if prior_statistic not in {"median", "reference_value"}:
		raise ValueError("high_percentile_eb_prior must be 'median' or 'reference_value'.")

	# (1) flagged fraction the incoming curve implies at each horizon
	q_raw = np.mean(risk_matrix >= high_curve[None, :], axis=0)

	# (2) which horizons have accrued enough events to count as informative
	supported = cumulative_events >= float(min_events_for_prior)
	if np.any(supported):
		reference_idx = int(np.argmax(supported))
	else:
		# never reaches min_events_for_prior inside this grid - fall back to
		# treating the last (best-supported) horizon as the sole reference
		reference_idx = n_times - 1
		supported = np.zeros(n_times, dtype=bool)
		supported[reference_idx] = True

	if prior_statistic == "median":
		prior_q = float(np.median(q_raw[supported]))
	else:
		prior_q = float(q_raw[reference_idx])
	prior_q = float(np.clip(prior_q, 0.0, 1.0))

	# (3) shrink towards that prior, weighted by accrued events
	alpha = cumulative_events / (cumulative_events + float(shrink_k))
	q_shrunk = alpha * q_raw + (1.0 - alpha) * prior_q
	if enforce_monotone:
		q_shrunk = np.maximum.accumulate(q_shrunk)
	q_shrunk = np.clip(q_shrunk, 0.0, 1.0)

	# (4) map the stabilised fraction back onto the risk scale. cutoff is the
	# smallest one honouring the fraction as a CAP - so ties in the risk column
	# can never turn e.g. a 9% target into flagging the whole cohort.
	shrunk_curve = np.array(
		[_cutoff_for_flagged_fraction(risk_matrix[:, j], q_shrunk[j]) for j in range(n_times)],
		dtype=float,
	)
	shrunk_curve = np.clip(shrunk_curve, 0.0, 1.0)
	# what the cutoffs actually deliver (<= q_shrunk wherever the risk column is
	# tied) - reported so that gap is visible instead of silently assumed away
	q_achieved = np.mean(risk_matrix >= shrunk_curve[None, :], axis=0)

	diagnostics = {
		"q_raw": q_raw,
		"q_shrunk": q_shrunk,
		"q_achieved": q_achieved,
		"alpha": alpha,
		"prior_q": prior_q,
		"reference_index": int(reference_idx),
		"cumulative_events": cumulative_events,
		"enforce_monotone": bool(enforce_monotone),
	}
	return shrunk_curve, diagnostics


def fit_dynamic_thresholds(
	estimator: object,
	x_train: pd.DataFrame,
	y_train: np.ndarray,
	time_points: Iterable[float],
	n_bootstrap: int = 200,
	low_percentile: float = 1.0,
	false_positive_cost: float = 1.0,
	false_negative_cost: Optional[float] = None,
	true_positive_cost: Optional[float] = None,
	true_negative_cost: float = 0.0,
	low_false_positive_cost: Optional[float] = None,
	low_false_negative_cost: Optional[float] = None,
	low_true_positive_cost: Optional[float] = None,
	low_true_negative_cost: Optional[float] = None,
	high_false_positive_cost: Optional[float] = None,
	high_false_negative_cost: Optional[float] = None,
	high_true_positive_cost: Optional[float] = None,
	high_true_negative_cost: Optional[float] = None,
	stratified_bootstrap: bool = False,
	max_time_bins_for_bootstrap_strata: int = 5,
	prevalence_method: str = "empirical_known",
	utility_weighting: str = "unweighted",
	threshold_estimation_method: str = "point",
	low_threshold_method: str = "roc_utility",
	intermediate_threshold_method: str = "youden_j",
	high_threshold_method: str = "roc_utility",
	curve_spline_postprocess: str = "none",
	curve_smoother: str = "spline",
	curve_rolling_mean_window: int = 121,
	curve_spline_variant: str = "restricted",
	curve_spline_quantiles: Optional[Tuple[float, ...]] = None,
	curve_spline_knots_within_grid: bool = False,
	high_curve_min_knot_time: Optional[float] = None,
	high_curve_zero_start_time: Optional[float] = None,
	high_curve_terminal_extrapolation: str = "linear",
	low_eb_shrink_k: float = 10.0,
	high_eb_shrink_k: float = 10.0,
	eb_support_counts: str = "future_events",
	high_percentile_eb: bool = False,
	high_percentile_eb_k: float = 10.0,
	high_percentile_eb_min_events: int = 10,
	high_percentile_eb_monotone: bool = True,
	high_percentile_eb_prior: str = "median",
	high_percentile_eb_order: str = "after_spline",
	prevalence_reference_time: Optional[float] = None,
	high_threshold_anchor_points: Optional[Dict[float, float]] = None,
	high_threshold_spline_quantiles: Tuple[float, ...] = (0.05, 0.275, 0.5, 0.725, 0.95),
	ipcw_min_censoring_survival: float = 1e-6,
	random_state: Optional[int] = 12,
	verbose: bool = False,
	log_every_bootstrap: Optional[int] = None,
) -> DynamicThresholdResult:
	"""Fit the model and derive dynamic low/high (and intermediate) risk thresholds per time point.

	t_low and t_high both come from the same ROC-utility optimization:
		U(c) = TPR(c) - m * FPR(c), threshold c maximizes U(c)
	where m = P(non-event)/P(event) * (C_FP - C_TN)/(C_FN - C_TP). Low and high can use
	different cost profiles via the low_*/high_* arguments; false_positive_cost etc. set
	the shared default whenever the specific ones aren't given.

	prevalence_method controls how event probability enters m: "kaplan_meier" uses
	1 - S(t) from a KM fit, "empirical_known" (default) just takes the observed event
	fraction among samples with known status at that horizon.

	utility_weighting="ipcw" applies inverse-probability-of-censoring weights to the ROC
	at each horizon, instead of the plain unweighted default.

	threshold_estimation_method="bootstrap_median" swaps the single-sample threshold for
	the median over n_bootstrap resamples (stratified by event indicator and a time bin
	when stratified_bootstrap=True, preserving stratum sizes). "point" is the default and
	just uses the plain single-sample threshold at each horizon - the historical behavior
	for t_high, and for t_low too unless one of the other low_threshold_method options below
	is picked.

	t_low has three modes (low_threshold_method): "roc_utility" is the optimization
	above (point or bootstrap-median depending on threshold_estimation_method);
	"bootstrap_percentile" instead takes the median, across bootstraps, of the
	low_percentile of the risk distribution at each horizon; "empirical_bayes" computes
	the roc_utility curve first and then EB-shrinks it towards a running minimum, with
	strength set by low_eb_shrink_k.

	t_intermediate (intermediate_threshold_method) defaults to Youden's J - "youden_j"
	(aliases "y_statistic", "youden", "y"), the c maximizing TPR(c) - FPR(c) - but can
	also just be the per-horizon mean risk ("mean_risk"/"mean") or median risk
	("median_risk"/"median").

	t_high (high_threshold_method) defaults to "roc_utility" as above. "empirical_bayes"
	shrinks that curve towards a running maximum instead (strength high_eb_shrink_k).
	The two spline options, "restricted_cubic_spline" and "cubic_spline", resample the
	curve at event-time quantile knots (default quantiles 0.05/0.275/0.5/0.725/0.95) and
	fit a natural or classic cubic spline through them - unless high_threshold_anchor_points
	is given as {time: threshold}, in which case those values override the knot values
	directly rather than being read off the ROC-utility curve.

	A separate additive post-smoothing stage sits on top of all of this, controlled by
	curve_spline_postprocess (default "none"; "high"/"low"/"intermediate" pick a single
	curve, "low_high" does two, "all" does all three) together with curve_smoother, which
	is the smoother actually applied. "spline" (default) is the event-quantile spline
	from above; "rolling_mean" instead takes a centred rolling mean over
	curve_rolling_mean_window days - less flexible, but it can't undershoot the curve or
	invent a discontinuity the way a spline can, and it doesn't extrapolate past the ends
	(the spline-only knobs below are ignored whenever this is picked). curve_spline_variant
	chooses "restricted" (natural spline, default) vs "normal" (classic cubic spline);
	curve_spline_quantiles overrides the knot quantiles, falling back to
	high_threshold_spline_quantiles when left unset; and curve_spline_knots_within_grid
	restricts knot placement to event times inside the modelled horizon, since events past
	the grid don't inform the shape being fitted there and, left in, tend to push the
	outer knots out and starve the late horizons of coverage.

	Three more knobs apply only to t_high within that smoothing stage:
	- high_curve_zero_start_time pins the day t_high actually leaves zero. Left unset,
	  that day falls wherever the default straight-line extrapolation below the first
	  knot happens to cross zero; set, the sub-knot segment becomes a quadratic that's
	  zero at that day and matches the spline's value and slope at the first knot, so
	  nothing at or after the first knot changes.
	- high_curve_min_knot_time drops spline knots placed before that day. Early horizons
	  often don't have enough accrued events for the per-horizon ROC optimum to mean much,
	  so this stops the spline from anchoring to them; the fit then uses whatever knots
	  survive and extrapolates backward from the earliest of them.
	- high_curve_terminal_extrapolation governs the segment above the last knot: "linear"
	  (default) continues the spline's right boundary slope out to the last event time,
	  "flat" just holds the last knot's value. Both count as extrapolation, since the last
	  knot already sits at the outermost quantile - but only "linear" can pull the curve
	  towards zero across exactly the late horizons where event accrual, and hence
	  anything downstream reading the curve, is strongest. Worth knowing before flipping it.

	The returned table is deliberately narrow - just time, t_low, t_intermediate, t_high,
	event_prevalence, m_value_low, m_value_high, roc_n_used, roc_computable.

	For old callers: false_positive_cost/false_negative_cost/true_positive_cost/
	true_negative_cost still set the legacy default cost profile, and n_bootstrap,
	stratified_bootstrap and max_time_bins_for_bootstrap_strata apply whenever any
	bootstrap-based method gets selected above.

	verbose=True prints stage timings and bootstrap progress with an ETA.
	"""
	if log_every_bootstrap is not None and log_every_bootstrap <= 0:
		raise ValueError("log_every_bootstrap must be > 0 when provided.")
	if prevalence_method not in {"kaplan_meier", "empirical_known"}:
		raise ValueError(
			"prevalence_method must be 'kaplan_meier' or 'empirical_known'."
		)
	if eb_support_counts not in {"future_events", "cumulative_events"}:
		raise ValueError(
			"eb_support_counts must be 'future_events' or 'cumulative_events'."
		)
	if high_percentile_eb_prior not in {"median", "reference_value"}:
		raise ValueError(
			"high_percentile_eb_prior must be 'median' or 'reference_value'."
		)
	if high_percentile_eb and float(high_percentile_eb_k) <= 0.0:
		raise ValueError("high_percentile_eb_k must be > 0 when high_percentile_eb is on.")
	if high_percentile_eb_order not in {"after_spline", "before_spline"}:
		raise ValueError("high_percentile_eb_order must be 'after_spline' or 'before_spline'.")
	if utility_weighting not in {"unweighted", "ipcw"}:
		raise ValueError("utility_weighting must be 'unweighted' or 'ipcw'.")
	if threshold_estimation_method not in {"point", "bootstrap_median"}:
		raise ValueError(
			"threshold_estimation_method must be 'point' or 'bootstrap_median'."
		)
	if low_threshold_method not in {"roc_utility", "bootstrap_percentile", "empirical_bayes"}:
		raise ValueError(
			"low_threshold_method must be 'roc_utility', 'bootstrap_percentile' or 'empirical_bayes'."
		)
	# map the user-facing aliases onto the internal method name
	_intermediate_aliases = {
		"youden_j": "youden_j",
		"y_statistic": "youden_j",
		"youden": "youden_j",
		"y": "youden_j",
		"mean_risk": "mean_risk",
		"mean": "mean_risk",
		"median_risk": "median_risk",
		"median": "median_risk",
	}
	intermediate_threshold_method = _intermediate_aliases.get(
		str(intermediate_threshold_method).strip().lower(),
		str(intermediate_threshold_method).strip().lower(),
	)
	if intermediate_threshold_method not in {"youden_j", "mean_risk", "median_risk"}:
		raise ValueError(
			"intermediate_threshold_method must be one of: "
			"'youden_j' (or 'y_statistic'), 'mean_risk' (or 'mean'), "
			"'median_risk' (or 'median')."
		)
	if high_threshold_method not in {"roc_utility", "empirical_bayes", "restricted_cubic_spline", "cubic_spline"}:
		raise ValueError(
			"high_threshold_method must be 'roc_utility', 'empirical_bayes', 'restricted_cubic_spline' or 'cubic_spline'."
		)
	if curve_spline_postprocess not in {"none", "high", "low", "intermediate", "low_high", "all"}:
		raise ValueError(
			"curve_spline_postprocess must be one of: "
			"'none', 'high', 'low', 'intermediate', 'low_high', 'all'."
		)
	if curve_spline_variant not in {"restricted", "normal"}:
		raise ValueError("curve_spline_variant must be 'restricted' or 'normal'.")
	if curve_smoother not in {"spline", "rolling_mean"}:
		raise ValueError("curve_smoother must be 'spline' or 'rolling_mean'.")
	if high_curve_terminal_extrapolation not in {"linear", "flat"}:
		raise ValueError(
			"high_curve_terminal_extrapolation must be 'linear' or 'flat'."
		)
	if high_curve_min_knot_time is not None:
		if float(high_curve_min_knot_time) < 0.0:
			raise ValueError("high_curve_min_knot_time must be >= 0.")
		if curve_spline_postprocess not in {"high", "low_high", "all"}:
			raise ValueError(
				"high_curve_min_knot_time requires curve_spline_postprocess to include "
				"the high curve ('high', 'low_high' or 'all')."
			)
	if not 0.0 <= float(low_percentile) <= 100.0:
		raise ValueError("low_percentile must be between 0 and 100.")
	if ipcw_min_censoring_survival <= 0.0:
		raise ValueError("ipcw_min_censoring_survival must be > 0.")
	if low_eb_shrink_k <= 0.0:
		raise ValueError("low_eb_shrink_k must be > 0.")
	if high_eb_shrink_k <= 0.0:
		raise ValueError("high_eb_shrink_k must be > 0.")
	bootstrap_needed = (
		threshold_estimation_method == "bootstrap_median"
		or low_threshold_method == "bootstrap_percentile"
	)
	if bootstrap_needed and int(n_bootstrap) < 1:
		raise ValueError("n_bootstrap must be >= 1 when threshold_estimation_method='bootstrap_median'.")
	if false_negative_cost is None:
		# old API named this true_positive_cost; keep reading it that way if given
		false_negative_cost = 1.0 if true_positive_cost is None else float(true_positive_cost)

	default_fp = float(false_positive_cost)
	default_fn = float(false_negative_cost)
	default_tp = 0.0 if true_positive_cost is None else float(true_positive_cost)
	default_tn = float(true_negative_cost)

	resolved_low_fp = default_fp if low_false_positive_cost is None else float(low_false_positive_cost)
	resolved_low_fn = default_fn if low_false_negative_cost is None else float(low_false_negative_cost)
	resolved_low_tp = default_tp if low_true_positive_cost is None else float(low_true_positive_cost)
	resolved_low_tn = default_tn if low_true_negative_cost is None else float(low_true_negative_cost)

	resolved_high_fp = default_fp if high_false_positive_cost is None else float(high_false_positive_cost)
	resolved_high_fn = default_fn if high_false_negative_cost is None else float(high_false_negative_cost)
	resolved_high_tp = default_tp if high_true_positive_cost is None else float(high_true_positive_cost)
	resolved_high_tn = default_tn if high_true_negative_cost is None else float(high_true_negative_cost)

	overall_start = perf_counter()
	if verbose:
		print("[fit_dynamic_thresholds] Starting threshold fitting...")

	rng = np.random.default_rng(random_state)
	time_points = np.asarray(list(time_points), dtype=float)
	if verbose:
		print(
			f"[fit_dynamic_thresholds] n_samples={x_train.shape[0]}, "
			f"n_time_points={time_points.shape[0]}"
		)
		if bootstrap_needed:
			print(
				f"[fit_dynamic_thresholds] bootstrap enabled with n_bootstrap={int(n_bootstrap)}"
			)
		print(f"[fit_dynamic_thresholds] low_threshold_method={low_threshold_method}")
		print(
			f"[fit_dynamic_thresholds] intermediate_threshold_method={intermediate_threshold_method}"
		)
		print(f"[fit_dynamic_thresholds] high_threshold_method={high_threshold_method}")
		if low_threshold_method == "bootstrap_percentile":
			print(f"[fit_dynamic_thresholds] low_percentile={float(low_percentile):.2f}")

	fit_start = perf_counter()
	base_model = clone(estimator)
	base_model.fit(x_train, y_train)
	if verbose:
		print(
			f"[fit_dynamic_thresholds] Base model fitted in "
			f"{perf_counter() - fit_start:.2f}s"
		)

	event_mask = _extract_event_indicator(y_train)
	time_values = _extract_time_values(y_train)
	# events still to come after t - largest early, smallest late, so it's actually
	# the WRONG support for the early-horizon instability problem. kept around only
	# because the older risk-space EB stage below was built around it.
	future_events_by_t = np.asarray(
		[int(np.sum(event_mask & (time_values > float(t)))) for t in time_points],
		dtype=float,
	)
	# events accrued by t - i.e. the positives the ROC at horizon t is actually
	# built from, so this is the support that determines how well-identified a
	# given horizon's threshold is
	cumulative_events_by_t = np.asarray(
		[int(np.sum(event_mask & (time_values <= float(t)))) for t in time_points],
		dtype=float,
	)
	eb_support_by_t = (
		cumulative_events_by_t if eb_support_counts == "cumulative_events"
		else future_events_by_t
	)
	km_survival_by_t = _km_survival_at_times(event_mask, time_values, time_points)
	km_event_prevalence_by_t = 1.0 - km_survival_by_t

	censoring_survival_at_sample = None
	censoring_survival_at_time = None
	if utility_weighting == "ipcw":
		# G(t), the censoring survival, estimated by running KM on the flipped event indicator
		censoring_event = ~event_mask
		censor_t, censor_surv = kaplan_meier_estimator(censoring_event, time_values)
		censoring_survival_at_sample = _step_survival_at_times(censor_t, censor_surv, time_values)
		censoring_survival_at_time = _step_survival_at_times(censor_t, censor_surv, time_points)
		censoring_survival_at_sample = np.clip(
			censoring_survival_at_sample,
			ipcw_min_censoring_survival,
			1.0,
		)
		censoring_survival_at_time = np.clip(
			censoring_survival_at_time,
			ipcw_min_censoring_survival,
			1.0,
		)

	global_event_prevalence = float(np.clip(km_event_prevalence_by_t[-1], 0.0, 1.0))
	# prevalence_reference_time freezes m at one horizon instead of recomputing it per
	# horizon. since m = (1-p)/p * (C_FP-C_TN)/(C_FN-C_TP), a per-horizon p makes m track
	# event accrual, so t_high's strictness picks up a time trend that's really about
	# accrual rather than clinical preference. fixing the reference removes that trend
	# and makes m purely a statement about the cost trade-off. None = old per-horizon behavior.
	reference_prevalence = None
	if prevalence_reference_time is not None:
		_ref_idx = int(np.argmin(np.abs(time_points - float(prevalence_reference_time))))
		if prevalence_method == "kaplan_meier":
			reference_prevalence = float(np.clip(km_event_prevalence_by_t[_ref_idx], 0.0, 1.0))
		else:
			_ref_t = float(time_points[_ref_idx])
			_ref_known = (time_values >= _ref_t) | (event_mask & (time_values <= _ref_t))
			reference_prevalence = (
				float(np.mean((event_mask[_ref_known] & (time_values[_ref_known] <= _ref_t)).astype(float)))
				if np.any(_ref_known) else global_event_prevalence
			)
	global_m_low = _compute_m_from_costs(
		event_prevalence=max(1e-6, min(1.0 - 1e-6, global_event_prevalence)),
		false_positive_cost=resolved_low_fp,
		false_negative_cost=resolved_low_fn,
		true_positive_cost=resolved_low_tp,
		true_negative_cost=resolved_low_tn,
	)
	global_m_high = _compute_m_from_costs(
		event_prevalence=max(1e-6, min(1.0 - 1e-6, global_event_prevalence)),
		false_positive_cost=resolved_high_fp,
		false_negative_cost=resolved_high_fn,
		true_positive_cost=resolved_high_tp,
		true_negative_cost=resolved_high_tn,
	)
	if verbose:
		print(
			f"[fit_dynamic_thresholds] global_event_prevalence={global_event_prevalence:.4f}, "
			f"global_m_low={global_m_low:.4f}, global_m_high={global_m_high:.4f}"
		)
		print(f"[fit_dynamic_thresholds] prevalence_method={prevalence_method}")
		print(f"[fit_dynamic_thresholds] utility_weighting={utility_weighting}")
		print(f"[fit_dynamic_thresholds] threshold_estimation_method={threshold_estimation_method}")

	risk_start = perf_counter()
	train_risk = _build_risk_matrix(base_model, x_train, time_points)
	if verbose:
		print(
			f"[fit_dynamic_thresholds] Built train risk matrix in "
			f"{perf_counter() - risk_start:.2f}s"
		)
	if not np.any(event_mask):
		raise ValueError("No event samples found in training data.")

	bootstrap_indices = None
	if bootstrap_needed:
		bootstrap_indices = []
		if stratified_bootstrap:
			bootstrap_strata = _build_bootstrap_strata_labels(
				events=event_mask,
				times=time_values,
				max_time_bins=int(max_time_bins_for_bootstrap_strata),
			)
		else:
			bootstrap_strata = None

		n_samples = int(x_train.shape[0])
		for _ in range(int(n_bootstrap)):
			if stratified_bootstrap:
				sample_idx = _stratified_bootstrap_indices(rng, bootstrap_strata)
			else:
				sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
			bootstrap_indices.append(sample_idx)

	rows = []
	roc_start = perf_counter()
	for time_idx, t_value in enumerate(time_points):
		# prevalence at this specific horizon, feeds into the m-value below
		known_mask_t = (time_values >= float(t_value)) | (
			event_mask & (time_values <= float(t_value))
		)
		if np.any(known_mask_t):
			events_by_t = event_mask[known_mask_t] & (time_values[known_mask_t] <= float(t_value))
			n_events_by_t = int(np.sum(events_by_t))
		else:
			events_by_t = np.asarray([], dtype=bool)
			n_events_by_t = 0

		if prevalence_method == "kaplan_meier":
			event_prevalence_t = float(np.clip(km_event_prevalence_by_t[time_idx], 0.0, 1.0))
		else:
			if np.any(known_mask_t):
				event_prevalence_t = float(np.mean(events_by_t.astype(float)))
			else:
				event_prevalence_t = global_event_prevalence

		sample_weights_t = None
		if utility_weighting == "ipcw":
			sample_weights_t = np.zeros_like(time_values, dtype=float)
			case_mask_t = event_mask & (time_values <= float(t_value))
			control_mask_t = time_values >= float(t_value)

			if np.any(case_mask_t):
				sample_weights_t[case_mask_t] = 1.0 / censoring_survival_at_sample[case_mask_t]
			sample_weights_t[control_mask_t] = 1.0 / float(censoring_survival_at_time[time_idx])

		# no events observed yet at this horizon -> force both thresholds to zero
		if n_events_by_t == 0:
			rows.append(
				{
					"time": float(t_value),
					"t_low": 0.0,
					"t_intermediate": 0.0,
					"t_high": 0.0,
					"m_value_low": np.nan,
					"m_value_high": np.nan,
					"roc_n_used": int(np.sum(known_mask_t)),
					"roc_computable": False,
					"event_prevalence": event_prevalence_t,
					"m_prevalence": np.nan,
				}
			)
			continue

		m_prevalence_t = (
			event_prevalence_t if reference_prevalence is None else reference_prevalence
		)
		event_prevalence_safe = max(1e-6, min(1.0 - 1e-6, m_prevalence_t))
		m_value_low_t = _compute_m_from_costs(
			event_prevalence=event_prevalence_safe,
			false_positive_cost=resolved_low_fp,
			false_negative_cost=resolved_low_fn,
			true_positive_cost=resolved_low_tp,
			true_negative_cost=resolved_low_tn,
		)
		m_value_high_t = _compute_m_from_costs(
			event_prevalence=event_prevalence_safe,
			false_positive_cost=resolved_high_fp,
			false_negative_cost=resolved_high_fn,
			true_positive_cost=resolved_high_tp,
			true_negative_cost=resolved_high_tn,
		)

		low_threshold_point_raw, low_n_used, low_u_max = _compute_threshold_from_roc_utility(
			scores=train_risk[:, time_idx],
			events=event_mask,
			times=time_values,
			time_point=float(t_value),
			m_value=m_value_low_t,
			sample_weights=sample_weights_t,
		)
		high_threshold_point_raw, high_n_used, high_u_max = _compute_threshold_from_roc_utility(
			scores=train_risk[:, time_idx],
			events=event_mask,
			times=time_values,
			time_point=float(t_value),
			m_value=m_value_high_t,
			sample_weights=sample_weights_t,
		)
		if intermediate_threshold_method == "youden_j":
			intermediate_threshold_raw, intermediate_n_used, intermediate_j_max = _compute_threshold_from_roc_youden(
				scores=train_risk[:, time_idx],
				events=event_mask,
				times=time_values,
				time_point=float(t_value),
				sample_weights=sample_weights_t,
			)
		elif intermediate_threshold_method == "mean_risk":
			# just the mean risk across patients at this horizon
			intermediate_threshold_raw = float(
				np.clip(np.mean(train_risk[:, time_idx]), 0.0, 1.0)
			)
			intermediate_n_used = int(train_risk.shape[0])
			intermediate_j_max = np.nan
		else:
			# ...and here the median instead
			intermediate_threshold_raw = float(
				np.clip(np.median(train_risk[:, time_idx]), 0.0, 1.0)
			)
			intermediate_n_used = int(train_risk.shape[0])
			intermediate_j_max = np.nan

		if low_threshold_method == "bootstrap_percentile":
			low_threshold_raw = float(
				np.clip(np.percentile(train_risk[:, time_idx], float(low_percentile)), 0.0, 1.0)
			)
		else:
			low_threshold_raw = low_threshold_point_raw
		high_threshold_raw = high_threshold_point_raw
		if bootstrap_needed:
			low_boot = []
			high_boot = []
			intermediate_boot = []
			for b_idx, sample_idx in enumerate(bootstrap_indices):
				weights_b = None if sample_weights_t is None else sample_weights_t[sample_idx]

				if low_threshold_method == "bootstrap_percentile":
					b_low = float(
						np.clip(
							np.percentile(train_risk[sample_idx, time_idx], float(low_percentile)),
							0.0,
							1.0,
						)
					)
					low_boot.append(b_low)
				else:
					b_low, _, _ = _compute_threshold_from_roc_utility(
						scores=train_risk[sample_idx, time_idx],
						events=event_mask[sample_idx],
						times=time_values[sample_idx],
						time_point=float(t_value),
						m_value=m_value_low_t,
						sample_weights=weights_b,
					)
					if not np.isnan(b_low):
						low_boot.append(float(np.clip(b_low, 0.0, 1.0)))

				if threshold_estimation_method == "bootstrap_median":
					b_high, _, _ = _compute_threshold_from_roc_utility(
						scores=train_risk[sample_idx, time_idx],
						events=event_mask[sample_idx],
						times=time_values[sample_idx],
						time_point=float(t_value),
						m_value=m_value_high_t,
						sample_weights=weights_b,
					)
					if not np.isnan(b_high):
						high_boot.append(float(np.clip(b_high, 0.0, 1.0)))

					if intermediate_threshold_method == "youden_j":
						b_intermediate, _, _ = _compute_threshold_from_roc_youden(
							scores=train_risk[sample_idx, time_idx],
							events=event_mask[sample_idx],
							times=time_values[sample_idx],
							time_point=float(t_value),
							sample_weights=weights_b,
						)
						if not np.isnan(b_intermediate):
							intermediate_boot.append(float(np.clip(b_intermediate, 0.0, 1.0)))
					elif intermediate_threshold_method == "mean_risk":
						b_intermediate = float(
							np.clip(np.mean(train_risk[sample_idx, time_idx]), 0.0, 1.0)
						)
						intermediate_boot.append(b_intermediate)
					else:
						b_intermediate = float(
							np.clip(np.median(train_risk[sample_idx, time_idx]), 0.0, 1.0)
						)
						intermediate_boot.append(b_intermediate)

				if (
					verbose
					and log_every_bootstrap is not None
					and (b_idx + 1) % int(log_every_bootstrap) == 0
					and time_idx % max(1, int(time_points.shape[0] // 10)) == 0
				):
					print(
						f"[fit_dynamic_thresholds] t={int(t_value)} bootstrap {b_idx + 1}/{int(n_bootstrap)}"
					)

			if len(low_boot) > 0:
				low_threshold_raw = float(np.median(low_boot))
			if len(high_boot) > 0:
				high_threshold_raw = float(np.median(high_boot))
			if len(intermediate_boot) > 0:
				intermediate_threshold_raw = float(np.median(intermediate_boot))

		if np.isnan(low_threshold_raw):
			low_threshold_raw = 0.0
		if np.isnan(high_threshold_raw):
			high_threshold_raw = 0.0
		if np.isnan(intermediate_threshold_raw):
			intermediate_threshold_raw = 0.5 * (float(low_threshold_raw) + float(high_threshold_raw))

		low_threshold_raw = float(low_threshold_raw)
		high_threshold_raw = float(high_threshold_raw)
		intermediate_threshold_raw = float(intermediate_threshold_raw)
		low_threshold = float(min(low_threshold_raw, high_threshold_raw))
		high_threshold = float(max(low_threshold_raw, high_threshold_raw))
		intermediate_threshold = float(np.clip(intermediate_threshold_raw, low_threshold, high_threshold))
		roc_n_used = int(max(low_n_used, high_n_used, intermediate_n_used))

		rows.append(
			{
				"time": float(t_value),
				"t_low": low_threshold,
				"t_intermediate": intermediate_threshold,
				"t_high": float(high_threshold),
				"m_value_low": m_value_low_t,
				"m_value_high": m_value_high_t,
				"roc_n_used": roc_n_used,
				"roc_computable": bool(
					(not np.isnan(low_u_max))
					or (not np.isnan(high_u_max))
					or (not np.isnan(intermediate_j_max))
				),
				"event_prevalence": event_prevalence_t,
				"m_prevalence": float(event_prevalence_safe),
			}
		)

	threshold_df = pd.DataFrame(rows)
	if low_threshold_method == "empirical_bayes":
		threshold_df["t_low"] = _empirical_bayes_shrink_curve(
			base_curve=threshold_df["t_low"].to_numpy(dtype=float),
			support_counts=eb_support_by_t,
			shrink_k=float(low_eb_shrink_k),
			direction="low",
		)
	if high_threshold_method == "empirical_bayes":
		threshold_df["t_high"] = _empirical_bayes_shrink_curve(
			base_curve=threshold_df["t_high"].to_numpy(dtype=float),
			support_counts=eb_support_by_t,
			shrink_k=float(high_eb_shrink_k),
			direction="high",
		)
	if high_threshold_method in {"restricted_cubic_spline", "cubic_spline"}:
		event_times_only = time_values[event_mask]
		spline_variant = "restricted" if high_threshold_method == "restricted_cubic_spline" else "normal"
		rcs_high_thresholds = _compute_high_thresholds_with_rcs(
			time_points=time_points,
			event_times=event_times_only,
			high_curve_values=threshold_df["t_high"].to_numpy(dtype=float),
			high_threshold_anchor_points=high_threshold_anchor_points,
			spline_quantiles=high_threshold_spline_quantiles,
			spline_variant=spline_variant,
		)
		threshold_df["t_high"] = np.maximum(
			threshold_df["t_low"].to_numpy(dtype=float),
			rcs_high_thresholds,
		)

	eb_percentile_diagnostics: Dict[str, object] = {}
	def apply_high_percentile_eb() -> None:
		nonlocal eb_percentile_diagnostics
		if not high_percentile_eb:
			return
		# the ordering is untested territory: "before_spline" deliberately runs the
		# additive smoother on top of the EB result, so at that point the curve no
		# longer exactly encodes q_high_shrunk.
		high_shrunk, eb_percentile_diagnostics = _empirical_bayes_shrink_flagged_fraction(
			risk_matrix=train_risk,
			high_curve=threshold_df["t_high"].to_numpy(dtype=float),
			cumulative_events=cumulative_events_by_t,
			shrink_k=float(high_percentile_eb_k),
			min_events_for_prior=int(high_percentile_eb_min_events),
			enforce_monotone=bool(high_percentile_eb_monotone),
			prior_statistic=str(high_percentile_eb_prior),
		)
		threshold_df["t_high"] = high_shrunk
		threshold_df["q_high_raw"] = eb_percentile_diagnostics["q_raw"]
		threshold_df["q_high_shrunk"] = eb_percentile_diagnostics["q_shrunk"]
		threshold_df["q_high_achieved"] = eb_percentile_diagnostics["q_achieved"]
		threshold_df["eb_alpha"] = eb_percentile_diagnostics["alpha"]
		threshold_df["cumulative_events"] = cumulative_events_by_t
		if verbose:
			_ref_j = int(eb_percentile_diagnostics["reference_index"])
			print(
				"[fit_dynamic_thresholds] flagged-fraction EB: prior q="
				f"{eb_percentile_diagnostics['prior_q']:.4f} from day "
				f"{float(time_points[_ref_j]):.0f} onwards "
				f"(>= {int(high_percentile_eb_min_events)} accrued events), "
				f"k={float(high_percentile_eb_k):.1f}, "
				f"order={high_percentile_eb_order}, "
				f"monotone={bool(high_percentile_eb_monotone)}"
			)

	if high_percentile_eb_order == "before_spline":
		apply_high_percentile_eb()
	if curve_spline_postprocess != "none":
		event_times_only = time_values[event_mask]
		if curve_spline_knots_within_grid:
			event_times_in_grid = event_times_only[event_times_only <= float(np.max(time_points))]
			if event_times_in_grid.size == 0:
				raise ValueError(
					"curve_spline_knots_within_grid=True but no event time falls inside "
					"the modelled horizon."
				)
			if verbose:
				print(
					"[fit_dynamic_thresholds] knot placement uses "
					f"{event_times_in_grid.size}/{event_times_only.size} events inside the "
					f"{float(np.max(time_points)):.0f}-day horizon"
				)
			event_times_only = event_times_in_grid
		if curve_spline_quantiles is None:
			spline_quantiles_to_use = high_threshold_spline_quantiles
		else:
			spline_quantiles_to_use = tuple(curve_spline_quantiles)

		if curve_spline_postprocess == "high":
			target_cols = ["t_high"]
		elif curve_spline_postprocess == "low":
			target_cols = ["t_low"]
		elif curve_spline_postprocess == "intermediate":
			target_cols = ["t_intermediate"]
		elif curve_spline_postprocess == "low_high":
			target_cols = ["t_low", "t_high"]
		else:
			target_cols = ["t_low", "t_intermediate", "t_high"]

		if verbose and high_curve_min_knot_time is not None:
			all_knot_days = np.unique(np.quantile(event_times_only, np.asarray(spline_quantiles_to_use, dtype=float)))
			kept_knot_days = all_knot_days[all_knot_days >= float(high_curve_min_knot_time)]
			print(
				"[fit_dynamic_thresholds] t_high knots "
				f"{np.round(all_knot_days, 0).tolist()} -> "
				f"{np.round(kept_knot_days, 0).tolist()} "
				f"(min_knot_time={float(high_curve_min_knot_time):.0f})"
			)

		for col in target_cols:
			if curve_smoother == "rolling_mean":
				threshold_df[col] = _rolling_mean_smooth_curve(
					threshold_df[col].to_numpy(dtype=float),
					curve_rolling_mean_window,
				)
				continue
			threshold_df[col] = _smooth_curve_with_event_quantile_spline(
				time_points=time_points,
				event_times=event_times_only,
				curve_values=threshold_df[col].to_numpy(dtype=float),
				spline_quantiles=spline_quantiles_to_use,
				spline_variant=curve_spline_variant,
				min_knot_time=high_curve_min_knot_time if col == "t_high" else None,
				zero_start_time=high_curve_zero_start_time if col == "t_high" else None,
				terminal_extrapolation=(
					high_curve_terminal_extrapolation if col == "t_high" else "linear"
				),
			)

	if high_percentile_eb_order == "after_spline":
		apply_high_percentile_eb()

	# whatever post-processing ran above could have crossed t_low and t_high, so
	# re-sort them pointwise one last time
	low_arr = threshold_df["t_low"].to_numpy(dtype=float)
	high_arr = threshold_df["t_high"].to_numpy(dtype=float)
	threshold_df["t_low"] = np.minimum(low_arr, high_arr)
	threshold_df["t_high"] = np.maximum(low_arr, high_arr)
	threshold_df["t_intermediate"] = np.clip(
		threshold_df["t_intermediate"].to_numpy(dtype=float),
		threshold_df["t_low"].to_numpy(dtype=float),
		threshold_df["t_high"].to_numpy(dtype=float),
	)
	risk_scores_train = pd.DataFrame(train_risk, columns=[f"risk_t_{int(t)}" for t in time_points])
	if verbose:
		print(
			f"[fit_dynamic_thresholds] ROC/high-threshold stage done in "
			f"{perf_counter() - roc_start:.2f}s"
		)
		print(
			f"[fit_dynamic_thresholds] Completed in {perf_counter() - overall_start:.2f}s"
		)

	return DynamicThresholdResult(
		thresholds=threshold_df,
		risk_scores_train=risk_scores_train,
		fitted_model=base_model,
		diagnostics={
			"future_events_by_t": future_events_by_t,
			"cumulative_events_by_t": cumulative_events_by_t,
			"eb_support_counts": eb_support_counts,
			"reference_prevalence": reference_prevalence,
			"percentile_eb": eb_percentile_diagnostics,
		},
	)


def assign_dynamic_risk_groups(
	risk_matrix: np.ndarray,
	threshold_df: pd.DataFrame,
	time_points: Iterable[float],
) -> pd.DataFrame:
	"""Label each sample low/intermediate/high at each time point, from the fitted thresholds."""
	time_points = list(time_points)
	labels = np.empty_like(risk_matrix, dtype=object)

	for j, t_value in enumerate(time_points):
		row = threshold_df.loc[threshold_df["time"] == float(t_value)]
		if row.empty:
			raise ValueError(f"Missing threshold row for time point {t_value}.")

		t_low = float(row.iloc[0]["t_low"])
		t_high = float(row.iloc[0]["t_high"])

		col_scores = risk_matrix[:, j]
		# strict boundaries - only strictly below t_low is "low", only strictly above
		# t_high is "high"; anything tied on a boundary falls into "intermediate"
		labels[:, j] = np.where(
			col_scores < t_low,
			"low",
			np.where(col_scores > t_high, "high", "intermediate"),
		)

	columns = [f"risk_group_t_{int(t)}" for t in time_points]
	return pd.DataFrame(labels, columns=columns)
