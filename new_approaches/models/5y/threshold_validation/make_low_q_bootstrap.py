"""Lean OOB bootstrap for the top_q_cohort high-risk rule at q in {0.05, 0.075}.

Reuses the exact data loading, model spec, stratified-resample scheme and
divergence guards from threshold_2.0_copy.ipynb cells 5/17/18, but skips the
full fit_dynamic_thresholds machinery (low/medium zone, ROC-utility curve,
splines) that the notebook computes and then discards under a top_q_cohort
high rule. Only c_high (a single percentile cutoff on risk at day 1825) is
needed for event capture, so only risk at day 1825 is evaluated -- not the
full 1825-day grid -- which is the main speedup over the notebook's loop.

Same model refit is reused for both q values within a replicate.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from joblib import Parallel, delayed

REPO_ROOT = os.path.abspath(r"C:\Users\Joao Goncalves\OneDrive\University\ist\Tese\thesis_2")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from functions import threshold as threshold_module  # noqa: E402

CONFIG = {
    "data_csv": os.path.join(REPO_ROOT, "new_approaches", "data", "data_cox", "data.csv"),
    "diagnosis_txt": os.path.join(REPO_ROOT, "new_approaches", "data", "data_cox", "diagnosis.txt"),
    "treatment_txt": os.path.join(REPO_ROOT, "new_approaches", "data", "data_cox", "treatment.txt"),
    "id_col": "PERSON_ID",
    "event_col": "Regional LN recurrence",
    "time_col": "Survival days",
    "administrative_cap_days": 1826.0,
    "continuous_features": [
        "Age", "Ki67 positive cells (%) at CB", "Conventional RT fraction",
        "Total administered dose", "Boost dose administered",
    ],
    "model_params": {"alphas": [0.000853168], "l1_ratio": 1.0, "fit_baseline_model": True},
    "random_seed": 42,
    "n_bootstrap": 500,
    "reference_horizon": 1825.0,
}
Q_VALUES = [0.05, 0.075]

_raw = pd.read_csv(CONFIG["data_csv"])
survival_dataset = _raw.drop(columns=[c for c in ["Unnamed: 0"] if c in _raw.columns])
with open(CONFIG["diagnosis_txt"], "r") as _f:
    diagnosis_variables = [line.strip() for line in _f if line.strip()]
with open(CONFIG["treatment_txt"], "r") as _f:
    treatment_variables = [line.strip() for line in _f if line.strip()]
feature_columns = treatment_variables + diagnosis_variables

_cap = float(CONFIG["administrative_cap_days"])
_beyond = survival_dataset[CONFIG["time_col"]] > _cap
survival_dataset.loc[_beyond, CONFIG["event_col"]] = 0
survival_dataset.loc[_beyond, CONFIG["time_col"]] = _cap

data_x = survival_dataset[feature_columns]
y = Surv.from_arrays(event=survival_dataset[CONFIG["event_col"]], time=survival_dataset[CONFIG["time_col"]])

_scaler = MinMaxScaler()
X = data_x.copy()
_cols_to_scale = [c for c in CONFIG["continuous_features"] if c in X.columns]
X[_cols_to_scale] = _scaler.fit_transform(X[_cols_to_scale])
X = X.reset_index(drop=True)

EVENT = y[y.dtype.names[0]].astype(bool)
TIME = y[y.dtype.names[1]].astype(float)
N_PATIENTS = int(X.shape[0])
REF_H = CONFIG["reference_horizon"]


def bootstrap_indices(rng, event):
    """Identical to the notebook's bootstrap_indices(stratify_on_event=True)."""
    event_rows = np.flatnonzero(event)
    censored_rows = np.flatnonzero(~event)
    return np.concatenate([
        rng.choice(event_rows, size=event_rows.size, replace=True),
        rng.choice(censored_rows, size=censored_rows.size, replace=True),
    ])


def top_q_cutoff(ref_col_inbag, q):
    n = ref_col_inbag.shape[0]
    k = int(np.clip(np.ceil((1.0 - float(q)) * n), 0, n - 1))
    boundary_row = int(np.argsort(ref_col_inbag, kind="mergesort")[k])
    return float(ref_col_inbag[boundary_row])


def run_replicate(replicate_index, seed):
    started = time.perf_counter()
    try:
        rng = np.random.default_rng(seed)
        inbag = bootstrap_indices(rng, EVENT)
        inbag_unique = np.unique(inbag)
        oob = np.setdiff1d(np.arange(N_PATIENTS), inbag_unique, assume_unique=True)
        if oob.size == 0:
            raise ValueError("empty out-of-bag set")

        model_b = CoxnetSurvivalAnalysis(**CONFIG["model_params"])
        model_b.fit(X.iloc[inbag], y[inbag])

        lp_inbag = np.asarray(model_b.predict(X.iloc[inbag]), dtype=float)
        if not np.all(np.isfinite(lp_inbag)):
            raise ValueError("non-finite linear predictor (model fit diverged)")

        risk_inbag_ref = threshold_module._build_risk_matrix(model_b, X.iloc[inbag], [REF_H])[:, 0]
        risk_oob_ref = threshold_module._build_risk_matrix(model_b, X.iloc[oob], [REF_H])[:, 0]

        _flat_rows = int(np.sum(risk_inbag_ref <= 0.0))
        if 0 < _flat_rows < risk_inbag_ref.shape[0]:
            raise ValueError("partial risk collapse (model fit diverged for a subset)")
        if float(np.ptp(risk_inbag_ref)) <= 0.0:
            raise ValueError("degenerate in-bag risk (model fit diverged): zero spread at reference horizon")

        event_oob = EVENT[oob]
        time_oob = TIME[oob]
        den = int((event_oob & (time_oob <= REF_H)).sum())

        out = {"replicate": int(replicate_index), "ok": True, "seed": int(seed),
               "n_inbag_unique": int(inbag_unique.size), "n_oob": int(oob.size),
               "n_oob_events_by_5y": den}
        for q in Q_VALUES:
            cutoff = top_q_cutoff(risk_inbag_ref, q)
            flagged_oob = risk_oob_ref >= cutoff
            num = int((event_oob & (time_oob <= REF_H) & flagged_oob).sum())
            ec = (num / den) if den > 0 else float("nan")
            out[f"q{q}_cutoff"] = cutoff
            out[f"q{q}_n_flagged_oob"] = int(flagged_oob.sum())
            out[f"q{q}_ec_num"] = num
            out[f"q{q}_ec_den"] = den
            out[f"q{q}_ec"] = ec
        out["elapsed"] = time.perf_counter() - started
        return out
    except Exception as exc:
        return {"replicate": int(replicate_index), "ok": False, "seed": int(seed),
                "error": f"{type(exc).__name__}: {exc}", "elapsed": time.perf_counter() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=CONFIG["n_bootstrap"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    B = args.n
    SEEDS = [int(s.generate_state(1)[0]) for s in
             np.random.SeedSequence(CONFIG["random_seed"]).spawn(B)]

    print(f"N_PATIENTS={N_PATIENTS}, events={int(EVENT.sum())}, running B={B} replicates, n_jobs=-1")
    t0 = time.perf_counter()
    results = []
    parallel = Parallel(n_jobs=-1, return_as="generator_unordered")
    completed = 0
    for payload in parallel(delayed(run_replicate)(b, SEEDS[b]) for b in range(B)):
        results.append(payload)
        completed += 1
        if completed % max(1, B // 20) == 0 or completed == B:
            elapsed = time.perf_counter() - t0
            eta = elapsed / completed * (B - completed)
            print(f"  {completed}/{B}  elapsed {elapsed/60:.2f} min  ETA {eta/60:.2f} min", flush=True)

    df = pd.DataFrame(results)
    out_path = args.out or os.path.join(os.path.dirname(__file__), f"q_bootstrap_B{B}.csv")
    df.to_csv(out_path, index=False)
    n_ok = int(df["ok"].sum())
    print(f"\ndone in {(time.perf_counter()-t0)/60:.2f} min. ok={n_ok}/{B} failed={B-n_ok}")
    print(f"saved -> {out_path}")

    ok = df[df["ok"]]
    for q in Q_VALUES:
        vals = ok[f"q{q}_ec"].dropna().to_numpy()
        med = np.median(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"q={q*100:g}%  OOB median EC(5y) = {100*med:.1f}%  95% CI [{100*lo:.1f}, {100*hi:.1f}]  "
              f"(n_valid_replicates={vals.size})")
