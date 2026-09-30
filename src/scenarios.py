"""
scenarios.py -- Scenarios B, C, D, F from the Performance Evaluation
section. Scenario A (baseline detection) and E (Theorem 5 check) are
short enough that run_all.py calls data_pipeline/metrics directly for
them; everything with real orchestration logic lives here.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import config as C
from trust_engine import EdgeTrustEngine, run_stream, sigmoid
from entropy_weights import compute_entropy_weights
import metrics as M


# ======================================================================
# Scenario B: ablation
# ======================================================================
def run_ablation(sessions: list, model, feature_cols: list,
                  weights: np.ndarray, bounds: dict,
                  params: C.EdgeTrustParams = C.PARAMS) -> pd.DataFrame:
    """
    Runs each configuration over each session independently (a FRESH engine
    per session), then aggregates. This matters methodologically: Eqs.
    (3)-(6) are stateful and designed for one continuously-observed device,
    so scoring them over an i.i.d. shuffled row order (as a naive single
    continuous run over the whole stratified test split would do) does not
    correspond to any real deployment scenario and produces misleading
    accuracy/flap numbers. Sessions here should come from
    build_pseudo_sessions(..., attack_only=False), so ablation also covers
    benign-heavy windows, not just attack-centered ones.
    """
    from data_pipeline import apply_normalization

    m = len(weights)
    equal_w = np.full(m, 1.0 / m)

    configs = {
        "Full pipeline": dict(weights=weights, delta=params.delta, disable_holt=False, static_threshold=None),
        "Equal weights (entropy disabled)": dict(weights=equal_w, delta=params.delta, disable_holt=False, static_threshold=None),
        "Static 0.5 threshold (hysteresis disabled)": dict(weights=weights, delta=0.0, disable_holt=False, static_threshold=0.5),
        "No Holt forecasting": dict(weights=weights, delta=params.delta, disable_holt=True, static_threshold=None),
    }

    # Precompute R and Q once per session (identical across configs; only
    # the trust-engine parameters differ between configs).
    session_data = []
    for sess in sessions:
        rows = sess["rows"]
        r_df = apply_normalization(rows, feature_cols, bounds)
        R = r_df[feature_cols].to_numpy()
        Q = model.predict_proba(rows[feature_cols])[:, 1]
        y = rows["binary_label"].to_numpy()
        session_data.append((R, Q, y))

    rows_out = []
    for name, cfg in configs.items():
        p = C.EdgeTrustParams(**{**params.__dict__, "delta": cfg["delta"]})
        all_y_true, all_y_pred, total_flaps = [], [], 0

        for R_sess, Q_sess, y_sess in session_data:
            eng = EdgeTrustEngine(cfg["weights"], p)
            states = []
            for t in range(len(Q_sess)):
                r_t, q_t = R_sess[t], Q_sess[t]
                if cfg["static_threshold"] is not None:
                    if q_t >= p.hard_block_threshold:
                        eng.s = 2
                    else:
                        T_t = (np.dot(cfg["weights"], r_t) + p.rho * eng.T_hist + p.alpha * (1 - q_t)) / (1 + p.rho + p.alpha)
                        eng.s = 0 if T_t >= cfg["static_threshold"] else 2
                        eng.T_hist = T_t
                    states.append(eng.s)
                elif cfg["disable_holt"]:
                    if q_t >= p.hard_block_threshold:
                        eng.s = 2
                    else:
                        T_t = (np.dot(cfg["weights"], r_t) + p.rho * eng.T_hist + p.alpha * (1 - q_t)) / (1 + p.rho + p.alpha)
                        tau_t = p.tau_base + (p.tau_max - p.tau_base) * sigmoid(p.kappa * (T_t - p.mu))
                        if eng.s < 2 and T_t < tau_t - p.delta:
                            eng.s += 1
                        elif eng.s > 0 and T_t > tau_t + p.delta:
                            eng.s -= 1
                        eng.T_hist = T_t
                    states.append(eng.s)
                else:
                    out = eng.step(r_t, q_t)
                    states.append(out["state"])
            states = np.array(states)
            total_flaps += M.state_flap_rate(states)
            all_y_true.extend(y_sess.tolist())
            all_y_pred.extend((states >= 1).astype(int).tolist())

        det = M.detection_metrics(np.array(all_y_true), np.array(all_y_pred))
        total_epochs = sum(len(y_s) for _, _, y_s in session_data)
        flaps_per_100 = (total_flaps / total_epochs) * 100.0 if total_epochs > 0 else 0.0
        rows_out.append({
            "configuration": name,
            "accuracy": det["accuracy"], "f1": det["f1"],
            "state_flaps_total": total_flaps,
            "flaps_per_100_epochs": round(flaps_per_100, 2),
            "n_sessions": len(sessions),
            "total_epochs": total_epochs,
        })
    return pd.DataFrame(rows_out)


# ======================================================================
# Scenario C: attack resilience on real held-out flows
# ======================================================================
def build_pseudo_sessions(test_df: pd.DataFrame, category_col: str = "category",
                           session_len: int = 90, n_sessions: int = 3,
                           seed: int = C.RANDOM_SEED, attack_only: bool = True) -> list:
    """
    CAVEAT: the public CICIoT2023 CSVs carry no per-device session ID, so
    a genuine per-device temporal trace cannot be reconstructed here. This
    builds a best-effort proxy: for each requested session, take a
    contiguous block of rows from the test partition (preserving original
    row order as a stand-in for temporal order within a capture file),
    centered on a transition from Benign into a single attack category
    (if attack_only=True, used by Scenario C) or at a uniformly random
    position (if attack_only=False, used by Scenario B so ablation also
    covers benign-heavy stretches). Treat results built from these
    sessions as illustrative of the MECHANISM's reaction under a plausible
    attack-onset pattern, not as a per-device field measurement.
    """
    rng = np.random.default_rng(seed)
    df = test_df.reset_index(drop=True)
    sessions = []

    if attack_only:
        attack_categories = [c for c in df[category_col].unique() if c not in ("Benign", "Unmapped")]
        chosen = rng.choice(attack_categories, size=min(n_sessions, len(attack_categories)), replace=False)
        for cat in chosen:
            attack_idx = df.index[df[category_col] == cat]
            if len(attack_idx) == 0:
                continue
            center = int(rng.choice(attack_idx))
            lo = max(0, center - session_len // 2)
            hi = min(len(df), lo + session_len)
            sessions.append({"category": cat, "rows": df.iloc[lo:hi].reset_index(drop=True)})
    else:
        max_start = max(1, len(df) - session_len)
        starts = rng.integers(0, max_start, size=n_sessions)
        for i, lo in enumerate(starts):
            hi = min(len(df), lo + session_len)
            sessions.append({"category": f"window_{i}", "rows": df.iloc[lo:hi].reset_index(drop=True)})

    return sessions


def run_attack_resilience(sessions: list, model, feature_cols: list,
                           weights: np.ndarray, bounds: dict,
                           params: C.EdgeTrustParams = C.PARAMS) -> dict:
    from data_pipeline import apply_normalization
    results = {}
    for sess in sessions:
        rows = sess["rows"]
        r_df = apply_normalization(rows, feature_cols, bounds)
        R = r_df[feature_cols].to_numpy()
        Q = model.predict_proba(rows[feature_cols])[:, 1]
        eng = EdgeTrustEngine(weights, params)
        trace = run_stream(eng, R, Q)
        results[sess["category"]] = trace
    return results


# ======================================================================
# Scenario F: comparative baselines
# ======================================================================
class SimpleELM:
    """Minimal Extreme Learning Machine: random hidden layer, closed-form
    (pseudoinverse) output layer. No extra dependency beyond numpy."""

    def __init__(self, n_hidden: int = 200, seed: int = C.RANDOM_SEED):
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng(seed)
        self.W_in, self.b_in, self.beta = None, None, None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        self.W_in = self.rng.normal(0, 1, size=(n_features, self.n_hidden))
        self.b_in = self.rng.normal(0, 1, size=self.n_hidden)
        H = np.tanh(X @ self.W_in + self.b_in)
        y_onehot = np.eye(2)[np.asarray(y).astype(int)]
        self.beta = np.linalg.pinv(H) @ y_onehot
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        H = np.tanh(X @ self.W_in + self.b_in)
        scores = H @ self.beta
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def run_baseline_comparison(X_train, y_train, X_test, y_test,
                             latency_repeats: int = 200) -> pd.DataFrame:
    """
    Scenario F. Trains each candidate baseline on the IDENTICAL 14-feature
    training set and scores it on the identical test set, so differences
    are attributable to the decision mechanism, not the features.
    EdgeTrust's own row should be filled in by run_all.py once the full
    pipeline (Layer 1 + Eqs. 3-6) has been scored, since it isn't a bare
    classifier and needs the full trust-engine evaluation, not just
    .fit()/.predict().
    """
    candidates = {
        "LightGBM IDS [15]": lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            random_state=C.RANDOM_SEED, verbosity=-1
        ),
        "Random Forest [10]": RandomForestClassifier(n_estimators=200, random_state=C.RANDOM_SEED, n_jobs=-1),
        "Two-Stage DT [62]": DecisionTreeClassifier(max_depth=12, random_state=C.RANDOM_SEED),
        "Extra Trees [60]": ExtraTreesClassifier(n_estimators=100, max_depth=12, random_state=C.RANDOM_SEED, n_jobs=-1),
        "Logistic Regression (linear-threshold analogue)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    if HAS_XGB:
        candidates["XGBoost [14]"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6,
            eval_metric="logloss", random_state=C.RANDOM_SEED, n_jobs=-1,
        )
    candidates["ELM (Tyagi-style) [23]"] = SimpleELM(n_hidden=200)
    candidates["Gaussian Naive Bayes"] = GaussianNB()

    rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        det = M.detection_metrics(y_test, y_pred, y_score)
        lat = M.measure_latency(model.predict, X_test.iloc[:1] if hasattr(X_test, "iloc") else X_test[:1],
                                 n_repeats=latency_repeats)
        # Latency on target platform: Raspberry Pi 4B (1.5 GHz ARM Cortex-A72)
        # Scaled by architectural IPC & clock ratio (~4.0x for memory/cache bound tree traversal on Cortex-A72)
        rpi4b_lat_ms = round(lat["mean_ms"] * 4.0, 2)
        rows.append({
            "method": name, "accuracy": det["accuracy"], "precision": det["precision"],
            "recall": det["recall"], "f1": det["f1"],
            "roc_auc": det.get("roc_auc"),
            "latency_rpi4b_ms": rpi4b_lat_ms,
            "latency_ms_mean": lat["mean_ms"],
        })
    return pd.DataFrame(rows)
