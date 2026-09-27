"""
Competing-risks sensitivity check for the 5-year regional-LN-recurrence model.

The modelling pipeline treats every non-recurrence exit as independent censoring.
This quantifies what that costs, by comparing the naive 1 - Kaplan-Meier estimator
against the Aalen-Johansen cumulative incidence function on the same risk sets.

Two competing-event definitions get reported. (A) death only, an unambiguous
competing event. (B) death or distant mets -- a sensitivity check, since
metastatic disease doesn't physically preclude a later regional recurrence but
it does take the patient out of the surveillance population.

Run:  python new_approaches/models/5y/competing_risks_check.py
"""
import numpy as np
import pandas as pd

RAW = 'BreastCareCond.xlsx'
MATRIX = 'new_approaches/data/data_cox/data.csv'
COEFS = 'new_approaches/models/5y/deployed_model/coefficients.csv'
YEAR = 365.25
HORIZONS = (2, 5, 10)
N_BOOT = 2000
SEED = 20260825


# --------------------------------------------------------------------------
# Rebuild the analysis cohort exactly as new_data.ipynb does
# --------------------------------------------------------------------------
def build_cohort():
    d = pd.read_excel(RAW)
    d.rename(columns={'Istotype at CB': 'Isotype at CB'}, inplace=True)

    # date_of_diagnosis(): first non-null of these four, in this order
    diag_cols = ['Date of CB ', 'Date of 1st breast operation',
                 'Date of intervention on sentinel lymph node',
                 'Date of intervention (axilla)']
    for c in diag_cols:
        d[c] = pd.to_datetime(d[c], errors='coerce')
    d['Date of diagnosis'] = d[diag_cols].bfill(axis=1).iloc[:, 0]

    # 'Date of 1st regional recurrence' doubles as the last-follow-up date for
    # non-recurrers, which is why remove_zeros_na() can require it for everyone.
    d['Date of 1st regional recurrence'] = pd.to_datetime(
        d['Date of 1st regional recurrence'], errors='coerce')

    d = d[d['Date of 1st regional recurrence'].notna() & d['Date of diagnosis'].notna()].copy()
    d['T'] = (d['Date of 1st regional recurrence'] - d['Date of diagnosis']).dt.days
    d = d[d['T'] > 0]                             # tte_negative_patients()
    d = d[d['EBC'] == 1].reset_index(drop=True)   # filter_EBC()

    # competing-event clocks, each gated on its own indicator
    for c in ['Date of follow-up (if died date of death)', 'Date of 1st distant metastases']:
        d[c] = pd.to_datetime(d[c], errors='coerce')
    d['T_death'] = ((d['Date of follow-up (if died date of death)'] - d['Date of diagnosis'])
                    .dt.days.where(d['Patient status at follow-up'].eq(4)))
    d['T_dist'] = ((d['Date of 1st distant metastases'] - d['Date of diagnosis'])
                   .dt.days.where(d['Distant metastases'].eq(1)))
    return d


def recode(T, ev, T_comp):
    """(time, 0/1 event, competing time) -> (t_tilde, status in {0,1,2})."""
    t = np.where(np.isnan(T_comp), T, np.minimum(T, T_comp))
    status = np.where(ev & (np.isnan(T_comp) | (T <= T_comp)), 1,
                      np.where(~np.isnan(T_comp) & (T_comp <= T), 2, 0))
    return t.astype(float), status.astype(int)


# --------------------------------------------------------------------------
# The two estimators, on identical risk sets
# --------------------------------------------------------------------------
def naive_one_minus_km(t, status, horizon):
    """1 - KM(t), competing events treated as ordinary censoring.

    Estimates 1 - exp(-int lambda_1), i.e. the risk that WOULD be seen if the
    competing event could be removed. Always >= the true CIF.
    """
    S = 1.0
    for tk in np.unique(t[status == 1]):
        if tk > horizon:
            break
        S *= 1.0 - ((t == tk) & (status == 1)).sum() / (t >= tk).sum()
    return 1.0 - S


def aalen_johansen(t, status, horizon):
    """CIF for cause 1: F1(t) = sum_k S(t_{k-1}) * d1_k / n_k.

    S is the ALL-CAUSE Kaplan-Meier (both causes count as events). That factor
    is exactly what the naive estimator drops.
    """
    S, cif = 1.0, 0.0
    for tk in np.unique(t[status > 0]):
        if tk > horizon:
            break
        n = (t >= tk).sum()
        d1 = ((t == tk) & (status == 1)).sum()
        d2 = ((t == tk) & (status == 2)).sum()
        cif += S * d1 / n           # weight by survival BEFORE this time
        S *= 1.0 - (d1 + d2) / n    # then advance all-cause survival
    return cif


def all_cause_risk(t, status, horizon):
    S = 1.0
    for tk in np.unique(t[status > 0]):
        if tk > horizon:
            break
        S *= 1.0 - ((t == tk) & (status > 0)).sum() / (t >= tk).sum()
    return 1.0 - S


# --------------------------------------------------------------------------
# Stratified bootstrap for the gap
# --------------------------------------------------------------------------
def boot_gap(t, status, horizon, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for (1-KM) - CIF, resampling within status strata."""
    rng = np.random.default_rng(seed)
    idx_by = [np.flatnonzero(status == k) for k in (0, 1, 2)]
    out = []
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by if len(ix)])
        tb, sb = t[take], status[take]
        out.append(naive_one_minus_km(tb, sb, horizon) - aalen_johansen(tb, sb, horizon))
    return np.percentile(out, [2.5, 97.5])


# --------------------------------------------------------------------------
def main():
    d = build_cohort()
    T = d['T'].to_numpy(float)
    ev = d['Regional LN recurrence'].eq(1).to_numpy()
    print('cohort n = %d,  regional-LN-recurrence events = %d' % (len(d), ev.sum()))
    print('median follow-up among censored = %.2f y, max = %.1f y\n'
          % (np.median(T[~ev]) / YEAR, T.max() / YEAR))

    # The estimator the pipeline actually reports: 1 - KM on the analysis clock,
    # which never truncates at a competing event. For death the clock ends at the
    # date of death anyway; for distant metastasis the patient stays "at risk".
    pipe_status = np.where(ev, 1, 0)
    print('--- as implemented in the pipeline (no competing event anywhere) ---')
    for h_y in HORIZONS:
        print('    %2d y   1-KM = %6.3f%%'
              % (h_y, naive_one_minus_km(T, pipe_status, h_y * YEAR) * 100))
    print()

    defs = [
        ('A. death only', d['T_death'].to_numpy(float)),
        ('B. death or distant mets', np.fmin(d['T_death'].to_numpy(float),
                                             d['T_dist'].to_numpy(float))),
    ]

    for name, T_comp in defs:
        t, status = recode(T, ev, T_comp)
        print('--- %s ---' % name)
        print('    status counts  censored=%d  cause1=%d  cause2=%d'
              % ((status == 0).sum(), (status == 1).sum(), (status == 2).sum()))
        for h_y in HORIZONS:
            h = h_y * YEAR
            pipe = naive_one_minus_km(T, pipe_status, h)
            naive = naive_one_minus_km(t, status, h)
            cif = aalen_johansen(t, status, h)
            lo, hi = boot_gap(t, status, h)
            print('    %2d y   pipeline = %6.3f%%   1-KM(cens. at comp.) = %6.3f%%   '
                  'AJ CIF = %6.3f%%   gap = %+.3f pp [%+.3f, %+.3f]   rel = %+.1f%%   '
                  'vs pipeline = %+.1f%%   all-cause risk = %.1f%%'
                  % (h_y, pipe * 100, naive * 100, cif * 100, (naive - cif) * 100,
                     lo * 100, hi * 100, (naive / cif - 1) * 100,
                     (pipe / cif - 1) * 100, all_cause_risk(t, status, h) * 100))
        print()

    # ---- stratified by the deployed model's linear predictor -------------
    m = pd.read_csv(MATRIX, index_col=0)
    co = pd.read_csv(COEFS)
    co = co[co['coefficient'] != 0]
    lp = np.zeros(len(m))
    for _, r in co.iterrows():
        lp = lp + r['coefficient'] * m[r['feature']].astype(float).to_numpy()
    m['lp'] = lp

    j = (m[['PERSON_ID', 'Survival days', 'lp']]
         .merge(d[['PERSON_ID', 'T', 'Regional LN recurrence', 'T_death', 'T_dist']],
                left_on=['PERSON_ID', 'Survival days'], right_on=['PERSON_ID', 'T'])
         .drop_duplicates(subset=['PERSON_ID', 'Survival days']))
    print('--- stratified by deployed-model risk (merged n = %d, events = %d) ---'
          % (len(j), j['Regional LN recurrence'].sum()))

    Tj = j['T'].to_numpy(float)
    evj = j['Regional LN recurrence'].eq(1).to_numpy()
    Tc = np.fmin(j['T_death'].to_numpy(float), j['T_dist'].to_numpy(float))
    tj, sj = recode(Tj, evj, Tc)
    lpj = j['lp'].to_numpy()
    q75, q90, q95 = np.quantile(lpj, [0.75, 0.90, 0.95])
    h = 5 * YEAR

    pipe_sj = np.where(evj, 1, 0)
    print('    %-14s%6s%5s%11s%9s%9s%9s%10s'
          % ('stratum', 'n', 'ev', 'pipeline %', '1-KM %', 'AJ %', 'rel', 'comp<=5y'))
    for lab, mask in [('all', np.ones(len(j), bool)),
                      ('bottom 75%', lpj < q75), ('top 25%', lpj >= q75),
                      ('top 10%', lpj >= q90), ('top 5%', lpj >= q95)]:
        p = naive_one_minus_km(Tj[mask], pipe_sj[mask], h)
        a = naive_one_minus_km(tj[mask], sj[mask], h)
        b = aalen_johansen(tj[mask], sj[mask], h)
        nc = ((sj[mask] == 2) & (tj[mask] <= h)).sum()
        print('    %-14s%6d%5d%11.3f%9.3f%9.3f%+8.1f%%%10d'
              % (lab, mask.sum(), evj[mask].sum(), p * 100, a * 100, b * 100,
                 (p / b - 1) * 100, nc))


if __name__ == '__main__':
    main()
